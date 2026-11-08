#!/usr/bin/env python3
"""
Bee Chat - a ChatGPT-style web UI (Flask) for your own LLM.

Files needed (put them next to this app.py, or in ./models/, or pass paths with --tokenizer/--chat/--tool):
    bee_tokenizer.model
    sft_stage2.pt        (normal chat model)
    sft_tool_best.pt     (tool-calling model: calculator, weather, time, unit converter, web search, ...)

Install:
    pip install flask torch sentencepiece requests ddgs tzdata

Run:
    python app.py                    ->  http://127.0.0.1:5000
    python app.py --host 0.0.0.0     ->  reachable from other devices on your network
    python app.py --tunnel           ->  (Colab) prints a public https://....trycloudflare.com link
"""
import os, re, sys, json, math, time, ast, operator, argparse, threading, types, pickle, random, contextlib
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
import secrets, hmac
from datetime import timedelta, date

import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm
from flask import Flask, Response, request, jsonify, stream_with_context, session, redirect, send_file

SCRIPT_DIR = Path(__file__).resolve().parent
SEQ = 512

# =============================================================== model (same as your notebooks)
@dataclass
class BeeConfig:
    vocab_size: int = 32000
    d_model: int = 640
    n_layers: int = 14
    n_heads: int = 10
    n_kv_heads: int = 2
    ffn_hidden: int = 1728
    seq_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_weights: bool = True
    @property
    def head_dim(self): return self.d_model // self.n_heads

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__(); self.eps = eps; self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def precompute_rope(head_dim, max_seq_len, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_seq_len).float(), inv)
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, -1); return torch.cat([-x2, x1], -1)

def apply_rope(x, cos, sin):
    return x * cos[None, None].to(x.dtype) + rotate_half(x) * sin[None, None].to(x.dtype)

class GQAAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.n_rep = c.n_heads // c.n_kv_heads
        self.wq = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.wk = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.wv = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.wo = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)
        self.dropout = c.dropout
    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = k.repeat_interleave(self.n_rep, 1), v.repeat_interleave(self.n_rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.wo(o.transpose(1, 2).contiguous().view(b, t, -1))

class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False); self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class BeeBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.d_model, c.rms_norm_eps); self.attn = GQAAttention(c)
        self.ffn_norm = RMSNorm(c.d_model, c.rms_norm_eps); self.ffn = SwiGLU(c.d_model, c.ffn_hidden)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))

class BeeModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cfg = c
        self.token_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList([BeeBlock(c) for _ in range(c.n_layers)])
        self.final_norm = RMSNorm(c.d_model, c.rms_norm_eps)
        if not c.tie_weights:
            self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        cos, sin = precompute_rope(c.head_dim, c.seq_len, c.rope_theta)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
    def forward(self, input_ids, targets=None):
        t = input_ids.shape[1]
        x = self.token_emb(input_ids)
        cos, sin = self.rope_cos[:t], self.rope_sin[:t]
        for blk in self.blocks: x = blk(x, cos, sin)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_emb.weight) if self.cfg.tie_weights else self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

def forward_cached(model, ids, cache, pos0):
    """Incremental forward (KV cache). ids: (1,t) tensor starting at absolute position pos0. Returns logits of the last position."""
    t = ids.shape[1]
    x = model.token_emb(ids)
    cos, sin = model.rope_cos[pos0:pos0 + t], model.rope_sin[pos0:pos0 + t]
    for li, blk in enumerate(model.blocks):
        a = blk.attn
        h = blk.attn_norm(x)
        q = a.wq(h).view(1, t, a.n_heads, a.hd).transpose(1, 2)
        k = a.wk(h).view(1, t, a.n_kv, a.hd).transpose(1, 2)
        v = a.wv(h).view(1, t, a.n_kv, a.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache[li] is not None:
            k = torch.cat([cache[li][0], k], dim=2); v = torch.cat([cache[li][1], v], dim=2)
        cache[li] = (k, v)
        kk, vv = k.repeat_interleave(a.n_rep, 1), v.repeat_interleave(a.n_rep, 1)
        o = F.scaled_dot_product_attention(q, kk, vv, is_causal=(pos0 == 0 and t > 1))
        x = x + a.wo(o.transpose(1, 2).contiguous().view(1, t, -1))
        x = x + blk.ffn(blk.ffn_norm(x))
    x = model.final_norm(x[:, -1:])
    logits = F.linear(x, model.token_emb.weight) if model.cfg.tie_weights else model.lm_head(x)
    return logits[0, 0]

# =============================================================== tools
CALL_OPEN, CALL_CLOSE = '<tool_call>', '</tool_call>'
RESP_OPEN, RESP_CLOSE = '<tool_response>', '</tool_response>'

def jdump(o): return json.dumps(o, separators=(',', ':'), ensure_ascii=False)
def fmt_call(name, args): return CALL_OPEN + jdump({'name': name, 'arguments': args}) + CALL_CLOSE
def fmt_resp(obj): return RESP_OPEN + (obj if isinstance(obj, str) else jdump(obj)) + RESP_CLOSE

def parse_calls(text):
    calls = []
    for m in re.finditer(re.escape(CALL_OPEN) + r'(.*?)' + re.escape(CALL_CLOSE), text, re.S):
        try: obj = json.loads(m.group(1).strip())
        except Exception: continue
        if isinstance(obj, dict) and isinstance(obj.get('name'), str) and isinstance(obj.get('arguments', {}), dict):
            calls.append({'name': obj['name'], 'arguments': obj.get('arguments', {})})
    if not calls and CALL_OPEN in text:
        m = re.search(re.escape(CALL_OPEN) + r'\s*(\{.*\})', text, re.S)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict) and isinstance(obj.get('name'), str):
                    calls.append({'name': obj['name'], 'arguments': obj.get('arguments', {}) if isinstance(obj.get('arguments', {}), dict) else {}})
            except Exception: pass
    return calls

def tool_spec(name, desc, params):
    return {'name': name, 'description': desc, 'params': [{'name': n, 'type': t, 'required': r} for n, t, r in params]}

# exactly the specs the tool model was trained with
SYN = {t['name']: t for t in [
    tool_spec('calculator', 'Evaluate a math expression', [('expression', 'string', True)]),
    tool_spec('get_weather', 'Get the current weather for a city', [('city', 'string', True), ('unit', 'string', False)]),
    tool_spec('get_time', 'Get the current time in a timezone', [('timezone', 'string', True)]),
    tool_spec('convert_units', 'Convert a value from one unit to another', [('value', 'number', True), ('from_unit', 'string', True), ('to_unit', 'string', True)]),
    tool_spec('web_search', 'Search the web for current information', [('query', 'string', True)]),
    tool_spec('set_reminder', 'Set a reminder for the user', [('text', 'string', True), ('time', 'string', True)]),
    tool_spec('random_number', 'Pick a random integer between min and max', [('min', 'integer', True), ('max', 'integer', True)]),
    tool_spec('days_between', 'Count the days between two dates (YYYY-MM-DD)', [('start_date', 'string', True), ('end_date', 'string', True)]),
]}
DEFAULT_TOOLS = ['calculator', 'web_search']
TOOL_ICONS = {'calculator': '🧮', 'get_weather': '⛅', 'get_time': '🕒', 'convert_units': '📏', 'web_search': '🌐',
              'set_reminder': '⏰', 'random_number': '🎲', 'days_between': '📅'}

def sig(t):
    ps = ', '.join(f"{p['name']}{'' if p['required'] else '?'}: {p['type']}" for p in t['params'])
    return f"{t['name']}({ps}) - {t['description']}"

def system_text(tools):
    return ('You are Bee. Tools: ' + ' | '.join(sig(t) for t in tools) +
            '. To use a tool, reply only with ' + CALL_OPEN + '{"name":"tool_name","arguments":{...}}' + CALL_CLOSE +
            '. If no tool fits, answer normally.')

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}
_UN = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FN = {'sqrt': math.sqrt, 'abs': abs, 'round': round, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
       'log': math.log, 'log10': math.log10, 'min': min, 'max': max}

def _ev(n):
    if isinstance(n, ast.Expression): return _ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): return n.value
    if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
        a, b = _ev(n.left), _ev(n.right)
        if isinstance(n.op, ast.Pow) and abs(b) > 100: raise ValueError('exponent too large')
        return _BIN[type(n.op)](a, b)
    if isinstance(n, ast.UnaryOp) and type(n.op) in _UN: return _UN[type(n.op)](_ev(n.operand))
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FN and not n.keywords:
        return _FN[n.func.id](*[_ev(a) for a in n.args])
    if isinstance(n, ast.Name) and n.id in ('pi', 'e'): return getattr(math, n.id)
    raise ValueError('unsupported expression')

def calc(expression):
    expr = str(expression).replace('^', '**').replace('×', '*').replace('÷', '/').replace(',', '')
    r = _ev(ast.parse(expr.strip(), mode='eval'))
    if isinstance(r, float):
        r = int(r) if r.is_integer() and abs(r) < 1e15 else round(r, 6)
    return r

UNITS = {'length': {'m': 1, 'km': 1000, 'cm': 0.01, 'mm': 0.001, 'miles': 1609.344, 'feet': 0.3048, 'inches': 0.0254, 'yards': 0.9144},
         'mass': {'kg': 1, 'g': 0.001, 'pounds': 0.45359237, 'ounces': 0.028349523125},
         'volume': {'liters': 1, 'ml': 0.001, 'gallons': 3.785411784, 'cups': 0.2365882365}}
TEMPS = ['celsius', 'fahrenheit', 'kelvin']
DISPLAY = {'km': ['km', 'kilometers', 'kilometres', 'kilometer'], 'm': ['meters', 'm', 'meter', 'metres'], 'cm': ['cm', 'centimeters'],
           'mm': ['mm'], 'miles': ['miles', 'mile', 'mi'], 'feet': ['feet', 'ft', 'foot'], 'inches': ['inches', 'inch', 'in'],
           'yards': ['yards'], 'kg': ['kg', 'kilograms'], 'g': ['grams', 'g'], 'pounds': ['pounds', 'lbs', 'lb'],
           'ounces': ['ounces', 'oz'], 'liters': ['liters', 'litres', 'l'], 'ml': ['ml', 'milliliters'], 'gallons': ['gallons'],
           'cups': ['cups'], 'celsius': ['celsius', '°c', 'c'], 'fahrenheit': ['fahrenheit', '°f', 'f'], 'kelvin': ['kelvin', 'k']}
ALIASES = {a.lower(): canon for canon, lst in DISPLAY.items() for a in lst + [canon]}

def _clean_num(x):
    x = round(float(x), 4)
    return int(x) if x.is_integer() else x

def convert(value, from_unit, to_unit):
    f, t = ALIASES.get(str(from_unit).lower().strip()), ALIASES.get(str(to_unit).lower().strip())
    if f is None or t is None: raise ValueError('unknown unit')
    v = float(value)
    if f in TEMPS and t in TEMPS:
        c = v if f == 'celsius' else (v - 32) * 5 / 9 if f == 'fahrenheit' else v - 273.15
        return _clean_num(c if t == 'celsius' else c * 9 / 5 + 32 if t == 'fahrenheit' else c + 273.15)
    for cat in UNITS.values():
        if f in cat and t in cat: return _clean_num(v * cat[f] / cat[t])
    raise ValueError('incompatible units')

CITY_TZ = {'surat': 'Asia/Kolkata', 'mumbai': 'Asia/Kolkata', 'delhi': 'Asia/Kolkata', 'ahmedabad': 'Asia/Kolkata', 'india': 'Asia/Kolkata',
           'london': 'Europe/London', 'paris': 'Europe/Paris', 'berlin': 'Europe/Berlin', 'new york': 'America/New_York',
           'los angeles': 'America/Los_Angeles', 'chicago': 'America/Chicago', 'tokyo': 'Asia/Tokyo', 'sydney': 'Australia/Sydney',
           'dubai': 'Asia/Dubai', 'singapore': 'Asia/Singapore', 'moscow': 'Europe/Moscow', 'toronto': 'America/Toronto',
           'beijing': 'Asia/Shanghai', 'seoul': 'Asia/Seoul', 'cairo': 'Africa/Cairo', 'rome': 'Europe/Rome', 'madrid': 'Europe/Madrid',
           'utc': 'UTC'}

def impl_time(timezone):
    from zoneinfo import ZoneInfo
    tz = str(timezone).strip()
    try: z = ZoneInfo(tz)
    except Exception:
        z = ZoneInfo(CITY_TZ[tz.lower()]); tz = CITY_TZ[tz.lower()]
    now = dt.datetime.now(z)
    return {'timezone': tz, 'date': now.strftime('%Y-%m-%d'), 'time': now.strftime('%H:%M')}

_WCODE = {0: 'clear', 1: 'mostly clear', 2: 'partly cloudy', 3: 'cloudy', 45: 'foggy', 48: 'foggy', 51: 'drizzly', 53: 'drizzly', 55: 'drizzly',
          61: 'rainy', 63: 'rainy', 65: 'rainy', 80: 'rainy', 81: 'rainy', 82: 'rainy', 95: 'stormy', 96: 'stormy', 99: 'stormy'}

def impl_weather(city, unit='celsius'):
    import requests
    g = requests.get('https://geocoding-api.open-meteo.com/v1/search', params={'name': city, 'count': 1}, timeout=8).json()['results'][0]
    w = requests.get('https://api.open-meteo.com/v1/forecast', params={'latitude': g['latitude'], 'longitude': g['longitude'],
                     'current': 'temperature_2m,relative_humidity_2m,weather_code'}, timeout=8).json()['current']
    c = w['temperature_2m']
    out = {'city': city, 'condition': _WCODE.get(w['weather_code'], 'cloudy'), 'humidity': int(w['relative_humidity_2m'])}
    if str(unit).lower().startswith('f'): out['temp_f'] = round(c * 9 / 5 + 32)
    else: out['temp_c'] = round(c)
    return out

REMINDERS = []
def impl_reminder(text, time):
    REMINDERS.append({'text': text, 'time': time})
    return {'status': 'ok', 'text': text, 'time': time}

def _clip(s, n):
    s = re.sub(r'\s+', ' ', s or '').strip()
    return s if len(s) <= n else s[:n].rsplit(' ', 1)[0].rstrip(',;:') + '...'

def web_search(query):
    """Real web search. Returns (compact result the MODEL sees, sources list for the UI)."""
    items = []
    try:
        try: from ddgs import DDGS
        except ImportError: from duckduckgo_search import DDGS
        for r in DDGS().text(query, max_results=3) or []:
            items.append({'title': r.get('title', ''), 'url': r.get('href', ''), 'snippet': r.get('body', '')})
    except Exception:
        items = []
    if not items:                                           # fallback: Wikipedia (no key needed)
        try:
            import requests
            h = {'User-Agent': 'BeeChat/1.0 (personal project)'}
            s = requests.get('https://en.wikipedia.org/w/api.php', params={'action': 'query', 'list': 'search', 'srsearch': query,
                             'format': 'json', 'srlimit': 2}, headers=h, timeout=8).json()
            for hit in s['query']['search']:
                title = hit['title']
                r = requests.get('https://en.wikipedia.org/api/rest_v1/page/summary/' + quote(title.replace(' ', '_')), headers=h, timeout=8).json()
                items.append({'title': title, 'url': r.get('content_urls', {}).get('desktop', {}).get('page', ''), 'snippet': r.get('extract', '')})
        except Exception:
            pass
    items = [i for i in items if i['snippet']]
    if not items: return {'error': 'no results found'}, []
    # the model only has a 512-token window: give it at most 2 short snippets, in the format it was trained on
    compact = {'results': [{'snippet': _clip(i['snippet'], 170)} for i in items[:2]]}
    return compact, [{'title': i['title'], 'url': i['url']} for i in items[:3]]

def run_tool(call):
    """-> (result for the model, sources for the UI)"""
    name, a = call['name'], call.get('arguments', {}) or {}
    try:
        if name == 'calculator':    return {'result': calc(a['expression'])}, []
        if name == 'get_weather':   return impl_weather(a['city'], a.get('unit', 'celsius')), []
        if name == 'get_time':      return impl_time(a['timezone']), []
        if name == 'convert_units': return {'value': a['value'], 'from': a['from_unit'], 'to': a['to_unit'],
                                            'result': convert(a['value'], a['from_unit'], a['to_unit'])}, []
        if name == 'web_search':    return web_search(str(a['query']))
        if name == 'set_reminder':  return impl_reminder(a['text'], a['time']), []
        if name == 'random_number': return {'number': random.randint(int(a['min']), int(a['max']))}, []
        if name == 'days_between':  return {'days': (dt.date.fromisoformat(a['end_date']) - dt.date.fromisoformat(a['start_date'])).days}, []
        return {'error': f'unknown tool {name}'}, []
    except Exception as e:
        return {'error': (str(e) or type(e).__name__)[:100]}, []

# =============================================================== engine: tokenizer, checkpoints, generation
sp = None
device = torch.device('cpu')
USER = ASST = END = SYS = EOS = -1
BANNED = []
END_BEFORE_ASST = True          # <|user|> q <|end|> <|assistant|> ...  (read from the tool checkpoint when present)
MODELS = {}                     # 'chat' / 'tools' -> {'model', 'cache_ok', 'step', 'file'}
GEN_LOCK = threading.Lock()     # one generation at a time (small local model)

def autocast_ctx():
    if device.type != 'cuda': return contextlib.nullcontext()
    dtype = torch.bfloat16 if (torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(0)[0] >= 8) else torch.float16
    return torch.autocast('cuda', dtype=dtype)

def find_file(name, cli):
    if cli: return Path(cli)
    for p in [SCRIPT_DIR / name, SCRIPT_DIR / 'models' / name, Path.cwd() / name, Path('/content/drive/MyDrive/bee_data') / name,
              Path('/content/drive/MyDrive/bee_data/model/sft') / name, Path('/content/drive/MyDrive/bee_data/model/tool') / name]:
        if p.exists(): return p
    return None

def load_ckpt(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception:                                      # checkpoint pickled BeeConfig from a notebook's __main__
        class _U(pickle.Unpickler):
            def find_class(self, module, name):
                return BeeConfig if name == 'BeeConfig' else super().find_class(module, name)
        pm = types.ModuleType('pm'); pm.__dict__.update(pickle.__dict__); pm.Unpickler = _U
        return torch.load(path, map_location='cpu', pickle_module=pm, weights_only=False)

@torch.no_grad()
def cache_self_test(model):
    """KV-cache logits must match the plain forward; otherwise we fall back to the slow-but-safe path."""
    try:
        ids = [USER] + sp.encode('What is the capital of France and why is it famous?') + [ASST]
        with autocast_ctx():
            full = model(torch.tensor([ids], device=device))[0][0, -1].float()
            cache = [None] * len(model.blocks)
            forward_cached(model, torch.tensor([ids[:-1]], device=device), cache, 0)
            inc = forward_cached(model, torch.tensor([[ids[-1]]], device=device), cache, len(ids) - 1).float()
        cos = torch.nn.functional.cosine_similarity(full, inc, dim=0).item()
        ok = cos > 0.999 and int(full.argmax()) == int(inc.argmax())
        print(f'   kv-cache self-test: cosine {cos:.5f} -> {"OK (fast mode)" if ok else "MISMATCH (using slow mode)"}')
        return ok
    except Exception as e:
        print('   kv-cache self-test failed, using slow mode:', repr(e)[:120]); return False

def load_model(key, path):
    global END_BEFORE_ASST
    print(f'loading {key}: {path}')
    ck = load_ckpt(path)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    cfg = ck.get('cfg') if isinstance(ck, dict) else None
    if isinstance(cfg, dict): cfg = BeeConfig(**cfg)
    if not isinstance(cfg, BeeConfig): cfg = BeeConfig()
    sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    m = BeeModel(cfg).to(device)
    missing, _ = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if 'rope' not in k]
    assert not bad, f'weights missing in {Path(path).name}: {bad[:3]}'
    m.eval()
    if key == 'tools' and isinstance(ck, dict) and 'end_before_asst' in ck:
        END_BEFORE_ASST = bool(ck['end_before_asst'])
    MODELS[key] = {'model': m, 'cache_ok': cache_self_test(m), 'step': ck.get('step') if isinstance(ck, dict) else None, 'file': Path(path).name}

def init_engine(args):
    global sp, device, USER, ASST, END, SYS, EOS, BANNED
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tok = find_file('bee_tokenizer.model', args.tokenizer)
    assert tok and tok.exists(), 'bee_tokenizer.model not found (put it next to app.py or use --tokenizer)'
    sp = spm.SentencePieceProcessor(model_file=str(tok))
    def pid(t):
        i = sp.piece_to_id(t); assert i != sp.unk_id(), f'{t} not in tokenizer'; return i
    USER, ASST, END = pid('<|user|>'), pid('<|assistant|>'), pid('<|end|>')
    try: SYS = pid('<|system|>')
    except AssertionError: SYS = -1
    EOS = sp.eos_id()
    BANNED = [i for i in {USER, ASST, SYS, sp.pad_id(), sp.unk_id(), sp.bos_id()} if i is not None and i >= 0]
    print(f'device: {device} | tokenizer: {tok.name} | vocab {sp.get_piece_size()}')
    for key, name, cli in [('tools', 'sft_tool_best.pt', args.tool), ('chat', 'sft_stage2.pt', args.chat)]:
        p = find_file(name, cli)
        if p and p.exists():
            try: load_model(key, p)
            except Exception as e: print(f'!! could not load {name}: {e!r}')
        else:
            print(f'-- {name} not found, "{key}" model disabled')
    assert MODELS, 'no model checkpoints found!'

# =============================================================== prompts
def encode_messages(tools, msgs, gen_prompt=True):
    ids = []
    if tools: ids += [SYS] + sp.encode(system_text(tools))
    for m in msgs:
        if m['role'] in ('user', 'tool'):
            ids += [USER] + sp.encode(m['text']) + ([END] if END_BEFORE_ASST else [])
        else:
            ids += [ASST] + sp.encode(m['text']) + [END]
    if gen_prompt: ids.append(ASST)
    return ids

def fit_messages(tools, msgs, min_new=96):
    """Drop the oldest exchanges until prompt + min_new tokens fit into the 512-token window. -> (ids, used_msgs) or (None, None)"""
    groups = []
    for m in msgs:
        if m['role'] == 'user' or not groups: groups.append([m])
        else: groups[-1].append(m)
    start = 0
    while True:
        used = [m for g in groups[start:] for m in g]
        ids = encode_messages(tools, used)
        if len(ids) + min_new <= SEQ: return ids, used
        if start < len(groups) - 1: start += 1; continue
        # a single exchange is too long: shorten its first user text
        if used and used[0]['role'] == 'user' and len(used[0]['text']) > 60:
            groups[start][0] = {'role': 'user', 'text': used[0]['text'][: int(len(used[0]['text']) * 0.8)]}; continue
        return None, None

# =============================================================== generation (streaming, KV cache)
@torch.no_grad()
def gen_stream(entry, prompt_ids, max_new, temperature, top_k, top_p, rep_pen, tool_mode):
    """Yields the decoded text so far after every new token."""
    model = entry['model']; use_cache = entry['cache_ok']
    ids, out = list(prompt_ids), []
    cache = [None] * len(model.blocks) if use_cache else None
    cur, pos0, text = torch.tensor([ids], device=device), 0, ''
    for step in range(max_new):
        if len(ids) >= SEQ: break
        with autocast_ctx():
            if use_cache: lg = forward_cached(model, cur, cache, pos0)
            else: lg = model(torch.tensor([ids], device=device))[0][0, -1]
        lg = lg.float()
        in_call = tool_mode and CALL_OPEN in text and CALL_CLOSE not in text
        if BANNED: lg[BANNED] = -float('inf')
        if rep_pen != 1.0 and out and not in_call:
            idx = torch.tensor(sorted(set(out[-128:])), device=device)
            sel = lg[idx]; lg[idx] = torch.where(sel > 0, sel / rep_pen, sel * rep_pen)
        greedy = temperature <= 0.01 or (tool_mode and (step == 0 or in_call))
        if greedy:
            nxt = int(lg.argmax())
        else:
            lg = lg / temperature
            if top_k and top_k > 0:
                kth = torch.topk(lg, min(top_k, lg.numel())).values[-1]; lg[lg < kth] = -float('inf')
            probs = torch.softmax(lg, -1)
            if top_p and top_p < 1.0:
                sp_, si = torch.sort(probs, descending=True); cum = torch.cumsum(sp_, 0)
                sp_[(cum - sp_) > top_p] = 0; probs = torch.zeros_like(probs).scatter_(0, si, sp_)
            nxt = int(torch.multinomial(probs / probs.sum(), 1))
        if nxt == END or nxt == EOS: break
        ids.append(nxt); out.append(nxt)
        text = sp.decode(out)
        yield text
        if CALL_CLOSE in text: break
        if len(out) >= 24 and out[-8:] == out[-16:-8] == out[-24:-16]: break      # stuck in a loop
        cur, pos0 = torch.tensor([[nxt]], device=device), len(ids) - 1

# =============================================================== one chat turn (tool loop) -> NDJSON events
MAX_ROUNDS = 3
SEARCH_HINT = re.compile(
    r"\b(search|google|look\s?up|find out|find information|find info|browse|online|internet|web|news|latest|newest|recent|"
    r"today'?s|current|currently|right now|price of|stock|score|release date|who (is|was|are|won|created|invented|founded|made|wrote)|"
    r"when (did|was|is)|how much (does|is|are))\b", re.I)

def looks_like_search(text):
    return bool(SEARCH_HINT.search(text or ''))

def compress_history(msgs, keep):
    """Small model + 512 tokens: remember only the last `keep` exchanges, and for old ones keep just question + final answer
    (tool calls / tool results of old turns are dropped; the final answer already contains what mattered)."""
    groups = []
    for m in msgs:
        if m['role'] == 'user' or not groups: groups.append([m])
        else: groups[-1].append(m)
    cur, prev = groups[-1], (groups[:-1][-keep:] if keep > 0 else [])
    out = []
    for g in prev:
        fin = [m for m in g if m['role'] == 'assistant' and CALL_OPEN not in m['text']]
        if fin: out += [g[0], fin[-1]]
    return out + cur

def exact_tool_answer(call, res):
    """(value, fallback sentence) for tools whose result must appear verbatim in the answer, else None."""
    if not isinstance(res, dict) or 'error' in res: return None
    a, n = call.get('arguments', {}) or {}, call['name']
    if n == 'calculator' and 'result' in res:   return res['result'], f"{a.get('expression')} = {res['result']}."
    if n == 'convert_units' and 'result' in res: return res['result'], f"{a.get('value')} {a.get('from_unit')} = {res['result']} {a.get('to_unit')}."
    if n == 'days_between' and 'days' in res:   return res['days'], f"There are {res['days']} days between {a.get('start_date')} and {a.get('end_date')}."
    if n == 'random_number' and 'number' in res: return res['number'], f"Your random number is {res['number']}."
    return None

def clean_answer(t):
    t = re.sub(r'</?tool_(call|response)>', '', t).strip()
    return t or '(no answer)'

def chat_events(req):
    key = req.get('model', 'tools')
    if key not in MODELS:
        yield {'t': 'error', 'v': f'model "{key}" is not loaded'}; return
    entry = MODELS[key]
    tool_mode = key == 'tools'
    msgs = [{'role': m['role'], 'text': str(m['text'])} for m in req['messages']]
    msgs = compress_history(msgs, max(0, min(int(req.get('memory', 0)), 6)))
    names = [n for n in (req.get('tools') or []) if n in SYN] if tool_mode else []
    web = bool(req.get('web')) and tool_mode
    if web and 'web_search' not in names: names.append('web_search')
    tools = [SYN[n] for n in names]
    temperature = float(req.get('temperature', 1.5)); max_new = max(16, min(int(req.get('max_new', 200)), 400))
    user_text = msgs[-1]['text']
    new, n_tok, t0 = [], 0, time.time()

    with GEN_LOCK:
        yield {'t': 'status', 'v': 'Thinking...'}
        if web:                                              # "Web search" toggle: always search first, then let the model answer
            q = re.sub(r'\s+', ' ', msgs[-1]['text']).strip()[:160]
            call = {'name': 'web_search', 'arguments': {'query': q}}
            yield {'t': 'status', 'v': 'Searching the web...'}
            res, sources = run_tool(call)
            new += [{'role': 'assistant', 'text': fmt_call(call['name'], call['arguments'])}, {'role': 'tool', 'text': fmt_resp(res)}]
            msgs += new
            yield {'t': 'tool', 'name': call['name'], 'args': call['arguments'], 'result': res, 'sources': sources}

        final_text, last_ctx, exact = None, 0, []
        for rnd in range(MAX_ROUNDS):
            ids, used = fit_messages(tools, msgs)
            if ids is None:
                yield {'t': 'error', 'v': 'The conversation is too long for the 512-token context. Start a new chat.'}; return
            last_ctx = len(ids)
            budget = max(8, min(max_new if rnd or not tool_mode else max(max_new, 160), SEQ - len(ids) - 1))
            text = ''
            for text in gen_stream(entry, ids, budget, min(temperature, 0.3) if (rnd or web) else temperature, 40, 0.92, 1.12, tool_mode):
                n_tok += 1; req['_tok'] = n_tok
                v = text.lstrip()
                if tool_mode and v and (v.startswith('<tool_call') or '<tool_call>'.startswith(v)):
                    continue                                 # a tool call is being written: do not show raw JSON
                yield {'t': 'text', 'v': text}
            calls = parse_calls(text) if tool_mode else []
            if (calls and 'web_search' in names and any(c['name'] == 'web_search' for c in calls)
                    and (web or not looks_like_search(user_text))):
                # the small model over-uses web_search: ignore it unless the question really asks for the web
                names = [n for n in names if n != 'web_search']; tools = [SYN[n] for n in names]
                yield {'t': 'status', 'v': 'Thinking...'}
                continue
            if not calls:
                final_text = clean_answer(text); break
            calls = calls[:4]
            call_text = ''.join(fmt_call(c['name'], c['arguments']) for c in calls)
            results = []
            for c in calls:
                yield {'t': 'status', 'v': f"Using {c['name']}..."}
                if c['name'] in names: res, sources = run_tool(c)
                else: res, sources = {'error': f"tool {c['name']} is not enabled"}, []
                results.append(res)
                ex = exact_tool_answer(c, res)
                if ex: exact.append(ex)
                yield {'t': 'tool', 'name': c['name'], 'args': c['arguments'], 'result': res, 'sources': sources}
            pair = [{'role': 'assistant', 'text': call_text}, {'role': 'tool', 'text': ''.join(fmt_resp(r) for r in results)}]
            new += pair; msgs += pair
            yield {'t': 'status', 'v': 'Writing the answer...'}
        if final_text is None: final_text = 'I could not finish the tool steps. Please try again.'
        if exact and final_text:                              # the 80M model sometimes copies a number wrongly: trust the tool
            plain = final_text.replace(',', '')
            if not all(re.search(r'(?<![\d.])' + re.escape(str(v)) + r'(?![\d])', plain) for v, _ in exact):
                final_text = ' '.join(f for _, f in exact)
        new.append({'role': 'assistant', 'text': final_text})
        yield {'t': 'text', 'v': final_text}
        dur = max(time.time() - t0, 1e-3)
        yield {'t': 'done', 'messages': new, 'stats': {'tokens': n_tok, 'secs': round(dur, 1), 'tps': round(n_tok / dur, 1), 'ctx': last_ctx}}

# =============================================================== Flask
app = Flask(__name__)

# =============================================================== visitors, limits, logs, admin
DATA_PATH = SCRIPT_DIR / 'bee_data.json'                       # all logs + visitor usage live in this one JSON file
DAILY_TOKEN_LIMIT = int(os.environ.get('BEE_DAILY_TOKENS', 2000))      # tokens per visitor per window (name kept for env var)
WINDOW_SEC = 12 * 3600                                                  # window length: 12 h, starts at the visitor's first used token
ADMIN_USER = os.environ.get('BEE_ADMIN_USER', 'Hqut@628')

def _secret_file(name, make):
    f = SCRIPT_DIR / name
    if not f.exists(): f.write_text(make())
    return f.read_text().strip()

def admin_password():                                    # default below; override with the BEE_ADMIN_PASSWORD environment variable
    return os.environ.get('BEE_ADMIN_PASSWORD') or 'Hqut@628'

app.secret_key = _secret_file('.bee_secret', lambda: secrets.token_hex(32))
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax', PERMANENT_SESSION_LIFETIME=timedelta(days=365))

STORE = {'logs': [], 'usage': {}}
STORE_LOCK = threading.RLock()

def init_db():                                           # load the JSON file (creates it on the first saved chat)
    global STORE
    with STORE_LOCK:
        STORE = {'logs': [], 'usage': {}}
        if DATA_PATH.exists():
            try:
                d = json.loads(DATA_PATH.read_text(encoding='utf-8'))
                STORE = {'logs': d.get('logs', []), 'usage': d.get('usage', {})}
            except Exception:
                DATA_PATH.rename(DATA_PATH.with_suffix('.broken.json')); print('!! bee_data.json was unreadable, moved to bee_data.broken.json')

def _save():                                             # atomic write: never leaves a half-written file
    tmp = DATA_PATH.with_suffix('.tmp')
    tmp.write_text(json.dumps(STORE, ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(tmp, DATA_PATH)

def get_sid():                                           # anonymous visitor id kept in a signed cookie (no login)
    sid = session.get('sid')
    if not sid:
        sid = secrets.token_hex(8); session['sid'] = sid; session.permanent = True
    return sid

def usage_info(sid):
    now = time.time()
    with STORE_LOCK: r = STORE['usage'].get(sid)
    if not r or now - r['start'] >= WINDOW_SEC: return {'used': 0, 'limit': DAILY_TOKEN_LIMIT, 'reset_in': 0}
    return {'used': r['tokens'], 'limit': DAILY_TOKEN_LIMIT, 'reset_in': int(r['start'] + WINDOW_SEC - now)}

def add_usage(sid, n):                                   # the 12 h window starts at the visitor's first used token
    now = time.time()
    with STORE_LOCK:
        r = STORE['usage'].get(sid)
        if not r or now - r['start'] >= WINDOW_SEC: STORE['usage'][sid] = {'start': now, 'tokens': n}
        else: r['tokens'] += n
        for k in [k for k, v in STORE['usage'].items() if now - v['start'] > 2 * WINDOW_SEC]: del STORE['usage'][k]
        _save()

def log_chat(sid, model, q, a, tools, tokens, retry, status):
    with STORE_LOCK:
        STORE['logs'].append({'id': len(STORE['logs']) + 1, 'ts': time.time(), 'sid': sid, 'model': model, 'question': q, 'answer': a,
                              'tools': tools, 'tokens': tokens, 'retry': retry, 'status': status})
        _save()

def fmt_wait(sec): return f'{int(sec) // 3600}h {int(sec) % 3600 // 60}m'

FAILS = {}                                               # ip -> [count, first_time]  (login brute-force guard)

def is_admin(): return session.get('admin_until', 0) > time.time()

def admin_only(f):
    import functools
    @functools.wraps(f)
    def w(*a, **k):
        if is_admin(): return f(*a, **k)
        return (jsonify({'error': 'login required'}), 401) if request.path.startswith('/api/') else redirect('/login')
    return w

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        ip = request.remote_addr or '?'; n, t0 = FAILS.get(ip, [0, time.time()])
        if time.time() - t0 > 300: n, t0 = 0, time.time()
        if n >= 5: msg = 'Too many attempts. Try again in a few minutes.'
        elif (hmac.compare_digest(request.form.get('user', '').encode(), ADMIN_USER.encode())
              and hmac.compare_digest(request.form.get('password', '').encode(), admin_password().encode())):
            FAILS.pop(ip, None); session['admin_until'] = time.time() + 8 * 3600
            return redirect('/admin')
        else:
            FAILS[ip] = [n + 1, t0]; msg = 'Wrong username or password.'
    return Response(LOGIN_HTML.replace('__MSG__', msg), mimetype='text/html')

@app.get('/logout')
def logout():
    session.pop('admin_until', None); return redirect('/login')

@app.get('/admin')
@admin_only
def admin_page(): return Response(ADMIN_HTML, mimetype='text/html')

@app.get('/api/usage')
def api_usage(): return jsonify(usage_info(get_sid()))

@app.get('/api/admin/stats')
@admin_only
def admin_stats():
    today = date.today(); d0 = time.mktime(today.timetuple())
    w0 = d0 - today.weekday() * 86400; m0 = time.mktime(today.replace(day=1).timetuple())
    with STORE_LOCK: L = list(STORE['logs'])
    since = lambda t: [r for r in L if r['ts'] >= t]
    td = since(d0)
    out = {'today': len(td), 'week': len(since(w0)), 'month': len(since(m0)), 'total': len(L),
           'tokens_today': sum(r['tokens'] for r in td), 'retries_today': sum(1 for r in td if r['retry']),
           'visitors_today': len({r['sid'] for r in td}), 'visitors_total': len({r['sid'] for r in L}), 'limit': DAILY_TOKEN_LIMIT}
    for key, fmt, n in [('day', '%Y-%m-%d', 30), ('week', '%Y-W%W', 12), ('month', '%Y-%m', 12)]:
        g = {}
        for r in L:
            x = g.setdefault(time.strftime(fmt, time.localtime(r['ts'])), {'n': 0, 't': 0, 'r': 0, 'u': set()})
            x['n'] += 1; x['t'] += r['tokens']; x['r'] += 1 if r['retry'] else 0; x['u'].add(r['sid'])
        out['by_' + key] = [{'p': k, 'n': g[k]['n'], 't': g[k]['t'], 'r': g[k]['r'], 'u': len(g[k]['u'])} for k in sorted(g)[-n:]]
    return jsonify(out)

@app.get('/api/admin/logs')
@admin_only
def admin_logs():
    page = max(0, int(request.args.get('page', 0))); q = request.args.get('q', '').strip().lower()
    with STORE_LOCK: L = list(STORE['logs'])
    if q: L = [r for r in L if q in r['question'].lower() or q in r['answer'].lower() or q in r['sid'].lower()]
    L.reverse()
    return jsonify({'total': len(L), 'rows': L[page * 25:(page + 1) * 25]})

@app.get('/favicon.ico')
def favicon():
    return send_file(SCRIPT_DIR / 'favicon.ico', mimetype='image/x-icon', max_age=86400)

@app.get('/resume')
def resume():                                            # put JalpitParmar.pdf (or .docx) next to app.py
    for n in ('JalpitParmar.pdf', 'JalpitParmar.docx'):
        f = SCRIPT_DIR / n
        if f.exists(): return send_file(f, download_name=n, max_age=3600)
    return 'Resume file not found. Put JalpitParmar.pdf next to app.py.', 404

@app.get('/')
def index():
    get_sid()
    pf = SCRIPT_DIR / 'bee_pages.html'                     # Research + Benchmark pages (#research, #benchmark)
    pages = pf.read_text(encoding='utf-8') if pf.exists() else ''
    html = INDEX_HTML.replace('</head>', '<link rel="icon" href="/favicon.ico" sizes="any"></head>', 1)
    return Response(html.replace('</body>', pages + '</body>', 1), mimetype='text/html')

@app.get('/api/info')
def info():
    return jsonify({
        'device': str(device),
        'models': [{'id': 'tools', 'label': 'Bee Tools', 'sub': 'tool calling + web search', 'ok': 'tools' in MODELS},
                   {'id': 'chat', 'label': 'Bee Chat', 'sub': 'plain chat model', 'ok': 'chat' in MODELS}],
        'tools': [{'name': n, 'desc': t['description'], 'icon': TOOL_ICONS.get(n, '🔧')} for n, t in SYN.items()],
        'default_tools': DEFAULT_TOOLS, 'seq': SEQ})

@app.post('/api/chat')
def chat():
    try:
        req = request.get_json(force=True)
        msgs = req['messages']
        assert isinstance(msgs, list) and msgs and msgs[-1]['role'] == 'user', 'last message must be from the user'
        assert all(m['role'] in ('user', 'assistant', 'tool') and isinstance(m.get('text'), str) for m in msgs), 'bad messages'
        assert sum(len(m['text']) for m in msgs) < 40000, 'too much text'
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    sid = get_sid(); u = usage_info(sid)
    if u['used'] >= u['limit']:
        return jsonify({'error': f"Token limit reached ({u['limit']} tokens). Please come back in {fmt_wait(u['reset_in'])}.", 'limit': True, **u}), 429
    req['max_new'] = max(16, min(int(req.get('max_new', 200)), u['limit'] - u['used']))
    question, retry = msgs[-1]['text'], 1 if req.get('retry') else 0
    n_user = len(sp.encode(question))                      # billed tokens = your question + everything Bee generates (retries too)
    def stream():
        st = {'answer': '', 'tools': [], 'status': 'stopped', 'done': False}
        def record():
            tokens = n_user + req.get('_tok', 0)
            add_usage(sid, tokens)
            log_chat(sid, req.get('model'), question, st['answer'], ','.join(st['tools']), tokens, retry, st['status'])
            return usage_info(sid)
        try:
            for ev in chat_events(req):
                if ev['t'] == 'tool': st['tools'].append(ev['name'])
                elif ev['t'] == 'text': st['answer'] = ev['v']
                elif ev['t'] == 'error': st['status'] = 'error'; st['answer'] = st['answer'] or ev['v']
                elif ev['t'] == 'done':
                    st['answer'], st['status'] = ev['messages'][-1]['text'], 'ok'
                    ev['stats'].update(record()); st['done'] = True
                yield json.dumps(ev, ensure_ascii=False) + '\n'
        except Exception as e:
            st['status'], st['answer'] = 'error', st['answer'] or f'{type(e).__name__}: {e}'
            yield json.dumps({'t': 'error', 'v': f'{type(e).__name__}: {e}'}) + '\n'
        finally:
            if not st['done']: record()
    return Response(stream_with_context(stream()), mimetype='application/x-ndjson',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

def start_tunnel(port):                                    # optional: public link from Colab (no account needed)
    import subprocess, shutil, urllib.request
    exe = shutil.which('cloudflared') or '/tmp/cloudflared'
    if not os.path.exists(exe):
        print('downloading cloudflared...')
        urllib.request.urlretrieve('https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64', exe)
        os.chmod(exe, 0o755)
    proc = subprocess.Popen([exe, 'tunnel', '--url', f'http://127.0.0.1:{port}'], stderr=subprocess.PIPE, text=True)
    def pump():
        shown = False
        for line in proc.stderr:
            m = re.search(r'https://[-a-z0-9]+\.trycloudflare\.com', line)
            if m and not shown:
                shown = True; print('\n=================================\n PUBLIC LINK:', m.group(0), '\n=================================\n', flush=True)
    threading.Thread(target=pump, daemon=True).start()

def main():
    ap = argparse.ArgumentParser(description='Bee Chat web UI')
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--tokenizer'); ap.add_argument('--chat', help='path to sft_stage2.pt'); ap.add_argument('--tool', help='path to sft_tool_best.pt')
    ap.add_argument('--cpu', action='store_true'); ap.add_argument('--tunnel', action='store_true')
    args = ap.parse_args()
    init_engine(args)
    init_db()
    print(f'\nAdmin panel: /login   user: {ADMIN_USER}   (limit {DAILY_TOKEN_LIMIT} tokens/visitor/12h)')
    if args.tunnel: start_tunnel(args.port)
    print(f'\nBee Chat is running ->  http://{"127.0.0.1" if args.host == "0.0.0.0" else args.host}:{args.port}\n')
    app.run(host=args.host, port=args.port, threaded=True, debug=False)

# =============================================================== the web page (served at /)
INDEX_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
  <title>Bee AI — Your Smart Assistant for Any Tasks</title>
  <script src="https://cdn.tailwindcss.com/3.4.17"></script>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #eef0f6;
      --purple-split: #6200ff;
      --pink-split: #ff99df;
    }

    * {
      box-sizing: border-box;
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    body {
      margin: 0;
      background: radial-gradient(circle at 20% 20%, #f5f3ff 0%, #eef0f7 45%, #e6e9f2 100%);
      min-height: 100vh;
      color: #18181b;
      overflow-x: hidden;
      transition: background 0.4s ease;
    }

    /* Reference Image Split Horizon Test Mode */
    body.split-test-mode {
      background: linear-gradient(180deg, #5900ff 0%, #5900ff 50%, #ff9ce3 50%, #ff9ce3 100%) !important;
    }

    /* ==========================================================
       PREMIUM APPLE LIQUID GLASS EFFECT (Right Side Reference)
       ========================================================== */
    .apple-liquid-glass {
      position: relative;
      background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.42) 0%,
        rgba(255, 255, 255, 0.14) 45%,
        rgba(255, 190, 245, 0.18) 80%,
        rgba(140, 100, 255, 0.24) 100%
      );
      backdrop-filter: blur(14px) saturate(190%);
      -webkit-backdrop-filter: blur(14px) saturate(190%);
      backdrop-filter: url(#apple-liquid-refraction) blur(14px) saturate(190%) brightness(1.06);
      -webkit-backdrop-filter: url(#apple-liquid-refraction) blur(14px) saturate(190%) brightness(1.06);
      box-shadow:
        /* Outer ambient depth */
        0 20px 45px -12px rgba(45, 20, 95, 0.16),
        0 4px 12px rgba(0, 0, 0, 0.05),
        /* 3D Curved Liquid Glass Rim & Inner Wall Refraction */
        inset 0 2.5px 3px 0 rgba(255, 255, 255, 0.95),
        inset 0 -3px 6px 0 rgba(255, 130, 230, 0.45),
        inset 3px 0 6px 0 rgba(255, 255, 255, 0.65),
        inset -3px 0 6px 0 rgba(130, 80, 255, 0.35),
        inset 0 0 18px 4px rgba(255, 255, 255, 0.28);
      border: 1px solid rgba(255, 255, 255, 0.55);
      isolation: isolate;
      transition: transform 0.28s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.28s ease;
    }

    /* Specular Meniscus Top-Left Edge Ring */
    .apple-liquid-glass::before {
      content: "";
      position: absolute;
      inset: 0;
      border-radius: inherit;
      padding: 1.6px;
      background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 1) 0%,
        rgba(255, 255, 255, 0.35) 35%,
        rgba(255, 150, 235, 0.5) 75%,
        rgba(255, 255, 255, 0.9) 100%
      );
      -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      pointer-events: none;
      z-index: 2;
    }

    /* Curved Liquid Meniscus Highlight Glint */
    .apple-liquid-glass::after {
      content: "";
      position: absolute;
      top: 3px;
      left: 8%;
      right: 8%;
      height: 36%;
      border-radius: 999px;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.55) 0%, rgba(255, 255, 255, 0) 100%);
      filter: blur(2px);
      pointer-events: none;
      z-index: 1;
    }

    .apple-liquid-pill {
      position: relative;
      background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.58) 0%,
        rgba(255, 255, 255, 0.22) 60%,
        rgba(245, 185, 255, 0.28) 100%
      );
      backdrop-filter: blur(12px) saturate(180%);
      -webkit-backdrop-filter: blur(12px) saturate(180%);
      box-shadow:
        0 8px 20px -4px rgba(55, 35, 110, 0.12),
        inset 0 2px 2px rgba(255, 255, 255, 0.95),
        inset 0 -2px 4px rgba(210, 120, 255, 0.35),
        inset 0 0 10px rgba(255, 255, 255, 0.35);
      border: 1px solid rgba(255, 255, 255, 0.65);
      transition: all 0.2s ease;
    }

    .apple-liquid-pill:hover {
      transform: translateY(-2px) scale(1.02);
      box-shadow:
        0 12px 24px -4px rgba(85, 50, 180, 0.2),
        inset 0 2px 3px rgba(255, 255, 255, 1),
        inset 0 -2px 5px rgba(235, 120, 255, 0.5);
    }

    /* 3D Bee Floating & Wing Flutter Animations */
    @keyframes beeFloat {
      0%, 100% { transform: translateY(0px) rotate(-1deg); }
      50% { transform: translateY(-12px) rotate(1.5deg); }
    }

    @keyframes wingFlutterLeft {
      0%, 100% { transform: rotate(0deg) scaleY(1); }
      50% { transform: rotate(-11deg) scaleY(0.92); }
    }

    @keyframes wingFlutterRight {
      0%, 100% { transform: rotate(0deg) scaleY(1); }
      50% { transform: rotate(11deg) scaleY(0.92); }
    }

    @keyframes shadowPulse {
      0%, 100% { transform: scale(1); opacity: 0.45; }
      50% { transform: scale(0.84); opacity: 0.28; }
    }

    .bee-mascot-float {
      animation: beeFloat 4.2s ease-in-out infinite;
    }

    .bee-wing-l {
      transform-origin: 175px 125px;
      animation: wingFlutterLeft 0.16s ease-in-out infinite;
    }

    .bee-wing-r {
      transform-origin: 225px 120px;
      animation: wingFlutterRight 0.16s ease-in-out infinite;
    }

    .bee-floor-shadow {
      animation: shadowPulse 4.2s ease-in-out infinite;
    }

    /* Ambient Colour Orbs behind Liquid Glass so refraction is visible */
    .ambient-refraction-blob {
      position: absolute;
      border-radius: 9999px;
      filter: blur(36px);
      pointer-events: none;
      z-index: 0;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-thumb {
      background: rgba(120, 110, 160, 0.25);
      border-radius: 999px;
    }
  
header.hdr{position:sticky;top:calc(env(safe-area-inset-top,0px) + .75rem)}
#chat-view{display:none;background:radial-gradient(circle at 20% 20%,#f5f3ff 0%,#eef0f7 45%,#e6e9f2 100%)}
#chat-view.open{display:flex}
#sendbtn{background:#5e5ce6;color:#fff;font-weight:700;font-size:13px;padding:10px 18px;border-radius:12px;display:flex;align-items:center;gap:8px;box-shadow:0 6px 14px -4px rgba(94,92,230,.5);transition:background .2s}
#sendbtn.stop{background:#18181b}
#sendbtn .sq{display:none;width:10px;height:10px;background:#fff;border-radius:2px}
#sendbtn.stop .sq{display:block}
.dots{display:inline-flex;gap:4px;vertical-align:middle;margin-left:3px}
.dots i{width:6px;height:6px;border-radius:50%;background:#6a5af0;animation:dot 1s infinite ease-in-out}
.dots i:nth-child(2){animation-delay:.15s}.dots i:nth-child(3){animation-delay:.3s}
@keyframes dot{0%,80%,100%{opacity:.25;transform:translateY(0)}40%{opacity:1;transform:translateY(-4px)}}

#web-chip.on{background:rgba(94,92,230,.2);color:#4340c9;border-color:#5e5ce6}
.chipb{padding:6px 12px;border-radius:999px;font-size:12px;font-weight:700;color:#3f3f46;display:inline-flex;gap:6px;align-items:center}
.step{margin:2px 0 8px;border-radius:14px;background:rgba(255,255,255,.55);border:1px solid rgba(255,255,255,.8);font-size:12.5px;overflow:hidden}
.step summary{cursor:pointer;padding:6px 10px;display:flex;gap:6px;align-items:center;list-style:none;color:#6b6b7b}
.step summary::-webkit-details-marker{display:none}.step summary b{color:#18181b}
.step summary code{flex:1;min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-size:11.5px}
.step .det{padding:2px 10px 10px}.step.bad summary b{color:#d33}
.step pre{margin:6px 0 0;background:rgba(255,255,255,.6);padding:8px;border-radius:10px;white-space:pre-wrap;word-break:break-word;font-size:11.5px}
.step a{display:block;color:#5e5ce6;font-size:12px;margin-top:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.out{word-break:break-word}.out pre{background:rgba(255,255,255,.6);padding:10px;border-radius:12px;overflow-x:auto;margin:8px 0;white-space:pre}
.out code{background:rgba(255,255,255,.6);padding:1px 5px;border-radius:6px;font-size:12.5px}.out pre code{background:none;padding:0}.out a{color:#5e5ce6}
.errt{color:#d33}.foot{font-size:11px;color:#8a8a99;margin-top:6px}.stat{font-size:12px;color:#6b6b7b;margin-left:4px}
#set-modal{display:none;background:rgba(30,20,60,.35)}#set-modal.open{display:flex}
.trow{display:flex;gap:10px;align-items:center;padding:6px 0;font-size:13.5px}.trow small{display:block;color:#71717a;font-size:11.5px}.trow label{flex:1}
.fld{display:flex;align-items:center;gap:10px;font-size:13.5px;padding:5px 0}.fld input[type=range]{flex:1;accent-color:#5e5ce6}.fld output{width:38px;text-align:right;color:#71717a}
#set-modal h3{font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:#71717a;margin:14px 0 4px}
.mopt.off{opacity:.4;pointer-events:none}
.acts{display:flex;gap:6px;margin-top:8px}.acts:empty{display:none}
.actb{padding:4px 11px;border-radius:999px;font-size:12px;font-weight:700;color:#52525b;display:inline-flex;gap:5px;align-items:center}
.actb.ok{color:#15803d}
</style>
</head>
<body class="min-h-screen">
<svg width="0" height="0" class="absolute pointer-events-none">
    <defs>
      <!-- True Optical Liquid Glass Displacement & Specular Filter -->
      <filter id="apple-liquid-refraction" x="-15%" y="-15%" width="130%" height="130%" color-interpolation-filters="sRGB">
        <feTurbulence type="fractalNoise" baseFrequency="0.012 0.018" numOctaves="2" seed="11" result="liquidWave" />
        <feGaussianBlur in="liquidWave" stdDeviation="6" result="smoothWave" />
        <feDisplacementMap in="SourceGraphic" in2="smoothWave" scale="16" xChannelSelector="R" yChannelSelector="G" result="refracted" />
        <feSpecularLighting in="smoothWave" surfaceScale="3" specularConstant="0.85" specularExponent="28" lighting-color="#ffffff" result="specularGlint">
          <fePointLight x="-180" y="-220" z="260" />
        </feSpecularLighting>
        <feComposite in="specularGlint" in2="SourceAlpha" operator="in" result="clippedGlint" />
        <feBlend mode="screen" in="refracted" in2="clippedGlint" />
      </filter>

      <!-- Reusable 3D Glowing Gradient Bee Mascot Symbol -->
      <g id="bee-3d-character">
        <!-- Translucent Back Wing -->
        <g class="bee-wing-r">
          <path d="M210 118 C242 54, 305 64, 294 114 C285 148, 232 142, 210 118 Z"
                fill="url(#wingGrad)" stroke="rgba(255,255,255,0.75)" stroke-width="1.8" />
        </g>

        <!-- Translucent Front Wing -->
        <g class="bee-wing-l">
          <path d="M182 115 C175 44, 246 34, 248 94 C249 128, 204 134, 182 115 Z"
                fill="url(#wingGradFront)" stroke="rgba(255,255,255,0.9)" stroke-width="2" />
          <!-- Wing vein highlight -->
          <path d="M196 105 Q218 72 235 78" stroke="rgba(255,255,255,0.65)" stroke-width="1.5" fill="none" stroke-linecap="round"/>
        </g>

        <!-- Cute Left Arm Waving -->
        <path d="M98 188 C66 168, 56 186, 75 208 C86 220, 102 214, 106 202 Z" fill="url(#armGrad)" />

        <!-- Back Stinger -->
        <path d="M294 218 L315 224 L292 238 Z" fill="#5e2a84" />

        <!-- Main 3D Chubby Bee Body (Clipped for Stripes) -->
        <g clip-path="url(#beeBodyClip)">
          <!-- Base Glowing Face/Belly Gradient -->
          <ellipse cx="192" cy="204" rx="105" ry="86" transform="rotate(-6 192 204)" fill="url(#beeBodyMain)" />

          <!-- Dark Violet Stripe 1 -->
          <path d="M205 102 C238 148, 236 235, 198 298 L234 298 C270 235, 272 148, 240 102 Z" fill="url(#stripeDarkGrad)" opacity="0.92"/>

          <!-- Warm Amber Gold Stripe -->
          <path d="M238 105 C268 152, 266 235, 232 296 L266 296 C296 235, 298 152, 270 105 Z" fill="url(#stripeGoldGrad)" />

          <!-- Back Deep Violet Stripe 2 -->
          <path d="M268 112 C294 156, 292 235, 264 294 L305 294 L305 112 Z" fill="url(#stripeDarkGrad)" />

          <!-- 3D Inner Spherical Rim Lighting (Apple-style 3D subsurface scattering) -->
          <ellipse cx="192" cy="204" rx="105" ry="86" transform="rotate(-6 192 204)" fill="url(#beeSubsurfaceGlow)" />
        </g>

        <!-- Cute Stubby Legs -->
        <path d="M148 278 C144 302, 164 304, 166 282 Z" fill="url(#legGrad)" />
        <path d="M204 282 C202 308, 224 308, 224 282 Z" fill="url(#legGrad)" />
        <path d="M250 272 C250 296, 270 294, 266 270 Z" fill="url(#legGradDark)" />

        <!-- Antennae -->
        <path d="M138 132 Q122 95 108 98" stroke="url(#antennaGrad)" stroke-width="6.5" stroke-linecap="round" fill="none" />
        <circle cx="106" cy="98" r="8" fill="url(#antennaTip)" />

        <path d="M178 126 Q178 86 194 84" stroke="url(#antennaGrad)" stroke-width="6.5" stroke-linecap="round" fill="none" />
        <circle cx="195" cy="84" r="8" fill="url(#antennaTip)" />

        <!-- Glowing Happy Curved Eyes (^ ^) -->
        <g filter="url(#eyeNeonGlow)">
          <path d="M116 186 Q128 168 140 186" stroke="#ffffff" stroke-width="6.5" stroke-linecap="round" fill="none" />
          <path d="M174 186 Q186 168 198 186" stroke="#ffffff" stroke-width="6.5" stroke-linecap="round" fill="none" />
        </g>

        <!-- Rosy Cheeks -->
        <ellipse cx="108" cy="202" rx="12" ry="7" fill="#ff4da6" opacity="0.45" filter="blur(3px)" />
        <ellipse cx="208" cy="202" rx="12" ry="7" fill="#ff4da6" opacity="0.45" filter="blur(3px)" />

        <!-- Cute Happy Mouth -->
        <path d="M146 196 Q157 212 168 196 Z" fill="#5b1947" />
        <path d="M150 202 Q157 208 164 202 Z" fill="#ff6b9d" />
      </g>

      <clipPath id="beeBodyClip">
        <ellipse cx="192" cy="204" rx="105" ry="86" transform="rotate(-6 192 204)" />
      </clipPath>

      <!-- Bee Gradients -->
      <radialGradient id="beeBodyMain" cx="32%" cy="38%" r="70%">
        <stop offset="0%" stop-color="#ff85e4" />
        <stop offset="38%" stop-color="#f24998" />
        <stop offset="72%" stop-color="#ff7a3d" />
        <stop offset="100%" stop-color="#8c38c8" />
      </radialGradient>

      <radialGradient id="beeSubsurfaceGlow" cx="28%" cy="22%" r="75%">
        <stop offset="0%" stop-color="rgba(255,255,255,0.62)" />
        <stop offset="35%" stop-color="rgba(255,255,255,0)" />
        <stop offset="82%" stop-color="rgba(85, 20, 140, 0.25)" />
        <stop offset="100%" stop-color="rgba(255, 220, 255, 0.45)" />
      </radialGradient>

      <linearGradient id="stripeDarkGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#8b42d6" />
        <stop offset="50%" stop-color="#5c248c" />
        <stop offset="100%" stop-color="#3f1668" />
      </linearGradient>

      <linearGradient id="stripeGoldGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ffd56b" />
        <stop offset="55%" stop-color="#ff8c42" />
        <stop offset="100%" stop-color="#d94e67" />
      </linearGradient>

      <linearGradient id="wingGradFront" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="rgba(255, 255, 255, 0.88)" />
        <stop offset="50%" stop-color="rgba(235, 215, 255, 0.55)" />
        <stop offset="100%" stop-color="rgba(195, 165, 250, 0.32)" />
      </linearGradient>

      <linearGradient id="wingGrad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="rgba(255, 255, 255, 0.65)" />
        <stop offset="100%" stop-color="rgba(200, 170, 255, 0.25)" />
      </linearGradient>

      <linearGradient id="armGrad" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#ff9478" />
        <stop offset="100%" stop-color="#d44286" />
      </linearGradient>

      <linearGradient id="legGrad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#b5448e" />
        <stop offset="100%" stop-color="#692378" />
      </linearGradient>

      <linearGradient id="legGradDark" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#692378" />
        <stop offset="100%" stop-color="#421254" />
      </linearGradient>

      <linearGradient id="antennaGrad" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0%" stop-color="#b83d98" />
        <stop offset="100%" stop-color="#7c2ae8" />
      </linearGradient>

      <radialGradient id="antennaTip" cx="35%" cy="35%" r="65%">
        <stop offset="0%" stop-color="#ff9ef5" />
        <stop offset="100%" stop-color="#7c2ae8" />
      </radialGradient>

      <filter id="eyeNeonGlow" x="-40%" y="-40%" width="180%" height="180%">
        <feGaussianBlur stdDeviation="3.5" result="blur" />
        <feMerge>
          <feMergeNode in="blur" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
  </svg>
<div id="landing" class="relative min-h-screen overflow-hidden pb-8">
  <div class="ambient-refraction-blob w-72 h-72 bg-purple-500/35 -top-10 -left-10"></div>
  <div class="ambient-refraction-blob w-80 h-80 bg-pink-400/40 top-1/3 right-0"></div>
  <div class="ambient-refraction-blob w-64 h-64 bg-amber-300/35 bottom-10 left-1/4"></div>
  <div class="ambient-refraction-blob w-64 h-64 bg-indigo-500/30 bottom-0 right-10"></div>

  <header class="hdr z-30 mx-3 sm:mx-6 lg:mx-auto max-w-6xl px-4 sm:px-5 py-2.5 rounded-2xl apple-liquid-glass flex items-center justify-between">
    <div class="flex items-center gap-2.5">
      <div class="w-9 h-9 rounded-xl apple-liquid-pill flex items-center justify-center"><svg class="w-7 h-7" viewBox="50 60 270 250"><use href="#bee-3d-character"/></svg></div>
      <span class="font-extrabold text-lg tracking-tight text-zinc-900">Bee</span>
      <span class="text-[11px] font-bold px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-700">1.0</span>
    </div>
    <nav class="hidden md:flex items-center gap-8 text-sm font-semibold text-zinc-700">
      <a href="#research" onclick="return false" class="hover:text-indigo-600 transition">Research</a>
      <a href="#benchmark" onclick="return false" class="hover:text-indigo-600 transition">Benchmark</a>
      <a href="#about" onclick="return false" class="hover:text-indigo-600 transition">About us</a>
      <a href="#privacy" onclick="return false" class="hover:text-indigo-600 transition">Privacy Policy</a>
    </nav>
    <div class="flex items-center gap-2">
      <button onclick="openChat()" class="px-4 py-2 rounded-full text-xs sm:text-sm font-bold bg-[#5a58e9] text-white shadow-md hover:bg-indigo-700 transition">Try Bee <svg class="inline w-3.5 h-3.5 -mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17L17 7M8 7h9v9"/></svg></button>
      <button onclick="toggleMenu()" aria-label="Menu" class="md:hidden apple-liquid-pill w-9 h-9 rounded-full text-zinc-800 flex items-center justify-center"><svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
    </div>
    <div id="mnav" style="display:none;position:absolute" class="absolute left-0 right-0 top-full mt-2 apple-liquid-glass rounded-2xl p-3 flex-col gap-1 text-sm font-semibold text-zinc-800">
      <a href="#research" onclick="return false" class="px-3 py-2 rounded-xl hover:bg-white/50">Research</a>
      <a href="#benchmark" onclick="return false" class="px-3 py-2 rounded-xl hover:bg-white/50">Benchmark</a>
      <a href="#about" onclick="return false" class="px-3 py-2 rounded-xl hover:bg-white/50">About us</a>
      <a href="#privacy" onclick="return false" class="px-3 py-2 rounded-xl hover:bg-white/50">Privacy</a>
    </div>
  </header>

  <main class="relative z-10 max-w-6xl mx-auto px-5 sm:px-8 pt-8 sm:pt-12 grid grid-cols-1 lg:grid-cols-2 gap-x-10 gap-y-4 content-center items-center text-center lg:text-left min-h-[calc(100vh-90px)]">
    <div class="lg:col-start-1 lg:row-start-1 lg:self-end">
      <h1 class="text-4xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight leading-[1.08] text-zinc-900">
        Your <span class="text-[#5e5ce6]">Smart</span><br><span class="text-[#5e5ce6]">Assistant</span> for<br>Any Tasks
      </h1>
    </div>
    <div class="lg:col-start-2 lg:row-start-1 lg:row-span-2 flex flex-col items-center relative">
      <div class="absolute top-0 left-4 sm:left-16 lg:left-6 z-10 apple-liquid-glass px-4 py-1.5 rounded-2xl rounded-bl-sm text-xs sm:text-sm font-bold text-zinc-800">Hello!</div>
      <div class="bee-mascot-float relative z-0">
        <svg class="w-56 h-56 sm:w-80 sm:h-80 lg:w-[26rem] lg:h-[26rem] drop-shadow-[0_24px_32px_rgba(165,70,220,0.32)]" viewBox="40 40 290 280"><use href="#bee-3d-character"/></svg>
      </div>
      <div class="bee-floor-shadow w-32 sm:w-44 h-6 rounded-full bg-gradient-to-r from-pink-400/50 via-purple-500/55 to-amber-400/50 blur-md -mt-3"></div>
    </div>
    <div class="lg:col-start-1 lg:row-start-2 lg:self-start flex flex-col items-center lg:items-start">
      <p class="text-sm sm:text-lg text-zinc-500 leading-relaxed mb-5 max-w-md">Get instant help and support<br>with any task or problem</p>
      <button onclick="openChat()" class="w-full max-w-xs sm:w-auto sm:max-w-none sm:px-14 py-4 rounded-2xl apple-liquid-glass font-bold text-sm sm:text-base text-zinc-900 flex items-center justify-center gap-2 hover:scale-[1.02] transition"><span>Get started</span><svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M13 6l6 6-6 6"/></svg></button>
    </div>
  </main>
  <footer class="relative z-10 text-center text-[12px] text-zinc-500 px-4 pt-2 pb-5" style="padding-bottom:calc(env(safe-area-inset-bottom,0px) + 1.25rem)">
    For learning purposes only. Bee is not for production use. <a href="#privacy" onclick="return false" class="font-bold text-indigo-600 hover:underline">Privacy Policy</a>
  </footer>
</div>

<div id="chat-view" class="fixed inset-0 z-40 flex-col overflow-hidden">
  <div class="ambient-refraction-blob w-64 h-64 bg-purple-500/35 top-20 -left-10"></div>
  <div class="ambient-refraction-blob w-64 h-64 bg-pink-400/40 bottom-24 -right-10"></div>
  <div class="relative z-10 flex flex-col h-full w-full max-w-3xl mx-auto px-3 sm:px-6">
    <div class="flex items-center justify-between pb-3 border-b border-zinc-200/60" style="padding-top:calc(env(safe-area-inset-top,0px) + .75rem)">
      <button onclick="closeChat()" class="apple-liquid-pill px-3 py-1.5 rounded-lg text-xs sm:text-sm font-bold text-zinc-700 flex items-center gap-1.5"><svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg>Back</button>
      <span class="text-sm font-extrabold text-zinc-900 flex items-center gap-1.5"><svg class="w-6 h-6" viewBox="50 60 270 250"><use href="#bee-3d-character"/></svg>Bee Chat</span>
      <div class="relative"><button id="model-btn" onclick="toggleModels(event)" class="apple-liquid-pill px-3 py-1.5 rounded-lg text-xs sm:text-sm font-bold text-indigo-700 flex items-center gap-1.5"><span id="model-name">Bee-1.0</span><svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg></button><div id="model-menu" style="display:none;position:absolute" class="right-0 top-full mt-2 w-60 apple-liquid-glass rounded-2xl p-2 z-50 flex-col gap-1"><button class="mopt bg-white/60 w-full text-left px-3 py-2 rounded-xl hover:bg-white/60 transition" data-m="Bee-1.0" onclick="pickModel(this)"><div class="text-sm font-bold text-zinc-900">Bee-1.0</div><div class="text-[11px] text-zinc-500">Default everyday model</div></button><button class="mopt w-full text-left px-3 py-2 rounded-xl hover:bg-white/60 transition" data-m="Bee-1.0 Lite" onclick="pickModel(this)"><div class="text-sm font-bold text-zinc-900">Bee-1.0 Lite</div><div class="text-[11px] text-zinc-500">Faster, lighter answers</div></button><button class="mopt w-full text-left px-3 py-2 rounded-xl hover:bg-white/60 transition" data-m="Bee-1.0 Tools" onclick="pickModel(this)"><div class="text-sm font-bold text-zinc-900">Bee-1.0 Tools</div><div class="text-[11px] text-zinc-500">Can call tools</div></button></div></div>
    </div>
    <div id="msgs" class="flex-1 overflow-y-auto space-y-3 py-4">
      <div id="welcome" class="flex flex-col items-center justify-center text-center" style="min-height:52vh">
        <svg class="w-20 h-20 bee-mascot-float drop-shadow-[0_14px_20px_rgba(165,70,220,0.3)]" viewBox="40 40 290 280"><use href="#bee-3d-character"/></svg>
        <h2 class="mt-3 text-3xl sm:text-4xl font-extrabold tracking-tight text-zinc-900">Hi, I'm <span class="text-[#5e5ce6]">Bee</span></h2>
        <p class="mt-1.5 text-sm text-zinc-500 max-w-sm">A tiny 80M-parameter LLM built from scratch. Try one of these or ask your own.</p>
        <div class="mt-5 flex flex-wrap justify-center gap-2 max-w-xl">
          <button class="chipb apple-liquid-pill" onclick="pickQ(this)">Who made you?</button>
          <button class="chipb apple-liquid-pill" onclick="pickQ(this)" data-t="1">Calculate 345 * 12</button>
          <button class="chipb apple-liquid-pill" onclick="pickQ(this)" data-t="1">Search the web for the latest AI news</button>
          <button class="chipb apple-liquid-pill" onclick="pickQ(this)">Tell me a short story about a bee</button>
          <button class="chipb apple-liquid-pill" onclick="pickQ(this)">Tell me a joke</button>
          <button class="chipb apple-liquid-pill" onclick="pickQ(this)">Write a short poem about rain</button>
        </div>
      </div>
    </div>
    <div class="flex items-center gap-2 mb-2 flex-wrap">
      <button id="web-chip" onclick="toggleWeb()" class="chipb apple-liquid-pill" title="ON = search the web for every message. OFF = Bee decides.">&#127760; Web search</button>
      <button onclick="openSettings()" class="chipb apple-liquid-pill">&#9881; Tools &amp; settings</button>
      <button onclick="newChat()" class="chipb apple-liquid-pill">&#65291; New chat</button>
      <span id="tok" class="chipb apple-liquid-pill" title="Free tokens per 12 hours. The window starts at your first message and resets 12 h later.">&#9889; <span id="tok-n">...</span><span style="display:inline-block;width:48px;height:5px;border-radius:9px;background:rgba(0,0,0,.1);overflow:hidden"><i id="tok-bar" style="display:block;height:100%;width:0;background:#5e5ce6;transition:width .4s"></i></span></span>
    </div>
    <div class="apple-liquid-glass p-2 pl-3 rounded-2xl flex items-center gap-2" style="margin-bottom:.5rem">
      <input id="chat-input" type="text" placeholder="Ask Bee..." autocomplete="off" class="flex-1 min-w-0 bg-transparent text-zinc-800 placeholder-zinc-400 focus:outline-none px-2 py-2" style="font-size:16px">
      <button id="sendbtn" onclick="send()"><span class="sq"></span><span id="sb-txt">Send</span></button>
    </div>
    <p class="text-center text-[11px] text-zinc-500" style="padding-bottom:calc(env(safe-area-inset-bottom,0px) + .5rem)">Bee is AI and can make mistakes. Chats are logged for the site admin.</p>
  </div>
</div>

<div id="set-modal" class="fixed inset-0 z-50 items-center justify-center p-4" onclick="if(event.target===this)closeSettings()">
 <div class="apple-liquid-glass rounded-3xl p-5 w-full max-w-md overflow-y-auto" style="background:rgba(255,255,255,.78);max-height:88vh">
  <div class="flex items-center justify-between"><h2 class="font-extrabold text-lg text-zinc-900">Tools &amp; settings</h2><button onclick="closeSettings()" class="apple-liquid-pill w-8 h-8 rounded-full font-bold">&#10005;</button></div>
  <p class="text-xs text-zinc-500 mt-1">The tool model decides by itself when to use a tool. Context window: 512 tokens, so keep chats short.</p>
  <div id="tool-box"><h3>Tools Bee may use</h3><div id="tool-list"></div></div>
  <h3>Generation</h3>
  <div class="fld">Creativity <input type="range" id="tempR" min="0" max="2" step="0.05"><output id="tempO"></output></div>
  <div class="fld">Max length <input type="range" id="maxR" min="32" max="400" step="8"><output id="maxO"></output></div>
  <div class="fld" title="0 = every question stands alone">Memory <input type="range" id="memR" min="0" max="4" step="1"><output id="memO"></output></div>
  <button onclick="closeSettings()" class="w-full mt-4 py-3 rounded-2xl font-bold text-white" style="background:#5e5ce6">Done</button>
 </div></div>
<script>

const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function md(t){
  const blocks = [];
  t = String(t).replace(/```(\w*)\n?([\s\S]*?)```/g, (m, l, c) => { blocks.push(c.replace(/\n$/, '')); return '\u0000' + (blocks.length - 1) + '\u0000'; });
  t = esc(t);
  t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>');
  t = t.replace(/(https?:\/\/[^\s<]+[^\s<.,;:)])/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  t = t.replace(/\n/g, '<br>');
  return t.replace(/\u0000(\d+)\u0000/g, (m, i) => '<pre><code>' + esc(blocks[+i]) + '</code></pre>');
}
const store={get(k,d){try{const v=localStorage.getItem(k);return v?JSON.parse(v):d}catch(e){return d}},set(k,v){try{localStorage.setItem(k,JSON.stringify(v))}catch(e){}}};
const landing=$('landing'),chat=$('chat-view'),msgs=$('msgs'),inp=$('chat-input'),btn=$('sendbtn');
const GREETING=msgs.innerHTML;
let INFO=null,busy=false,ctrl=null,conv=[];
let S=Object.assign({model:'tools',tools:null,web:false,temp:1.5,maxNew:200,memory:0},store.get('bee_settings',{}));
if(S.memV!==2){S.memory=0;S.memV=2}
if(S.tempV!==4){S.temp=1.5;S.tempV=4}
if(S.toolV!==1){S.tools=null;S.toolV=1}   // one-time reset: only calculator + web_search are on by default
S.web=false;
const save=()=>store.set('bee_settings',S);
const meta=n=>(INFO&&INFO.tools.find(t=>t.name===n))||{icon:'\u{1F527}',desc:''};
const BEE='<svg class="w-5 h-5" viewBox="50 60 270 250"><use href="#bee-3d-character"/></svg>';

function toggleModels(e){e.stopPropagation();const m=$('model-menu');m.style.display=m.style.display==='flex'?'none':'flex'}
function pickModel(b){if(busy)return;S.model=b.dataset.m;save();$('model-menu').style.display='none';applyUI()}
document.addEventListener('click',()=>{$('model-menu').style.display='none'});
function toggleMenu(){const m=$('mnav');m.style.display=m.style.display==='flex'?'none':'flex'}
function openChat(){landing.style.display='none';chat.classList.add('open');$('mnav').style.display='none';setTimeout(()=>inp.focus(),60)}
function closeChat(){if(ctrl)ctrl.abort();chat.classList.remove('open');landing.style.display='';scrollTo(0,0)}
function setBtn(stop){btn.classList.toggle('stop',stop);$('sb-txt').textContent=stop?'Stop':'Send'}
function newChat(){if(busy)return;conv=[];msgs.innerHTML=GREETING;inp.focus()}
function toggleWeb(){S.web=!S.web;save();applyUI()}
function openSettings(){buildSettings();$('set-modal').classList.add('open')}
function closeSettings(){$('set-modal').classList.remove('open')}

function applyUI(){
  if(!INFO)return;
  const m=INFO.models.find(x=>x.id===S.model)||INFO.models[0];
  $('model-name').textContent=m.label;
  $('model-menu').innerHTML=INFO.models.map(x=>'<button class="mopt'+(x.ok?'':' off')+(x.id===S.model?' bg-white/60':'')+' w-full text-left px-3 py-2 rounded-xl hover:bg-white/60 transition" data-m="'+x.id+'" onclick="pickModel(this)"><div class="text-sm font-bold text-zinc-900">'+esc(x.label)+'</div><div class="text-[11px] text-zinc-500">'+esc(x.ok?x.sub:'not loaded')+'</div></button>').join('');
  const tm=S.model==='tools';
  $('web-chip').style.display=tm?'inline-flex':'none';$('web-chip').classList.toggle('on',S.web&&tm);
}
function buildSettings(){
  const on=new Set(S.tools||INFO.default_tools);
  $('tool-list').innerHTML=INFO.tools.map(t=>'<div class="trow"><span>'+t.icon+'</span><label for="tl_'+t.name+'"><b>'+esc(t.name)+'</b><small>'+esc(t.desc)+'</small></label><input type="checkbox" id="tl_'+t.name+'" data-tool="'+t.name+'"'+(on.has(t.name)?' checked':'')+'></div>').join('');
  $('tool-box').style.display=S.model==='tools'?'block':'none';
  [['temp','tempR','tempO'],['maxNew','maxR','maxO'],['memory','memR','memO']].forEach(([k,r,o])=>{$(r).value=S[k];$(o).textContent=S[k];$(r).oninput=e=>{S[k]=+e.target.value;$(o).textContent=S[k];save()}});
  $('tool-list').onchange=()=>{S.tools=[...document.querySelectorAll('[data-tool]')].filter(x=>x.checked).map(x=>x.dataset.tool);save()};
}

function pickQ(b){if(busy||!INFO)return;if(b.dataset.t&&INFO.models.find(m=>m.id==='tools'&&m.ok)){S.model='tools';save();applyUI()}inp.value=b.textContent;send()}
function addUser(text){const w=$('welcome');if(w)w.remove();const d=document.createElement('div');d.className='flex justify-end';const b=document.createElement('div');b.className='apple-liquid-pill px-4 py-2 rounded-2xl rounded-br-sm text-sm font-semibold text-zinc-800 max-w-[85%] break-words';b.textContent=text;d.appendChild(b);msgs.appendChild(d)}
function addBee(){
  const d=document.createElement('div');d.className='apple-liquid-glass p-3.5 rounded-2xl rounded-bl-sm text-sm text-zinc-800 max-w-[85%] break-words';
  d.innerHTML='<strong class="inline-flex items-center gap-1 align-middle">'+BEE+'Bee:</strong><div class="steps"></div><div class="out"></div><span class="dots"><i></i><i></i><i></i></span><span class="stat"></span><div class="foot"></div><div class="acts"></div>';
  msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
  return {box:d,steps:d.querySelector('.steps'),out:d.querySelector('.out'),dots:d.querySelector('.dots'),stat:d.querySelector('.stat'),foot:d.querySelector('.foot'),acts:d.querySelector('.acts')};
}
function stepHTML(ev){
  const bad=ev.result&&ev.result.error;
  let det='<div>'+esc(meta(ev.name).desc)+'</div><pre>'+esc(JSON.stringify({arguments:ev.args,result:ev.result},null,2))+'</pre>';
  (ev.sources||[]).forEach(x=>{if(x.url)det+='<a target="_blank" rel="noopener" href="'+esc(x.url)+'">'+esc(x.title||x.url)+'</a>'});
  return '<details class="step'+(bad?' bad':'')+'"><summary><span>'+meta(ev.name).icon+'</span><b>'+esc(ev.name)+'</b><code>'+esc(JSON.stringify(ev.args))+'</code><span>'+(bad?'failed':'done')+'</span></summary><div class="det">'+det+'</div></details>';
}

function fmtWait(s){return Math.floor(s/3600)+'h '+Math.floor(s%3600/60)+'m'}
function setTok(u){const n=$('tok-n');if(!n||u.used===undefined)return;
  const out=u.used>=u.limit,left=Math.max(0,u.limit-u.used),pct=Math.min(100,u.used/u.limit*100);
  n.textContent=out?'Limit reached \u00b7 come back in '+fmtWait(u.reset_in):'Used '+u.used+' \u00b7 '+left+' left of '+u.limit;
  $('tok').style.color=out?'#d33':'';const bar=$('tok-bar');bar.style.width=pct+'%';bar.style.background=pct>=90?'#d33':pct>=70?'#e8a100':'#5e5ce6'}
function copyText(t){
  if(navigator.clipboard&&window.isSecureContext)return navigator.clipboard.writeText(t);
  return new Promise((ok,no)=>{const a=document.createElement('textarea');a.value=t;a.style.cssText='position:fixed;opacity:0';document.body.appendChild(a);a.select();
    try{document.execCommand('copy')?ok():no()}catch(e){no(e)}a.remove()});
}
function addActions(b,text,nNew,finalText){
  document.querySelectorAll('.retry').forEach(x=>x.remove());          // Retry only on the newest answer
  if(finalText){
    const c=document.createElement('button');c.className='actb apple-liquid-pill';c.innerHTML='&#128203; Copy';
    c.onclick=()=>copyText(finalText).then(()=>{c.innerHTML='&#10003; Copied';c.classList.add('ok');setTimeout(()=>{c.innerHTML='&#128203; Copy';c.classList.remove('ok')},1400)},()=>{c.textContent='Copy failed'});
    b.acts.appendChild(c);
  }
  const r=document.createElement('button');r.className='actb apple-liquid-pill retry';r.innerHTML='&#8635; Retry';
  r.onclick=()=>{if(busy)return;conv=conv.slice(0,conv.length-nNew);b.box.remove();run(text,true)};
  b.acts.appendChild(r);
}
async function send(){
  if(busy){if(ctrl)ctrl.abort();return}
  const text=inp.value.trim();if(!text||!INFO)return;
  addUser(text);inp.value='';
  await run(text);
}
async function run(text,isRetry){
  busy=true;setBtn(true);
  const b=addBee();let answer='',nNew=0,finalText='';
  const body={model:S.model,messages:conv.concat([{role:'user',text}]),tools:S.tools||INFO.default_tools,web:S.web&&S.model==='tools',temperature:S.temp,max_new:S.maxNew,memory:S.memory,retry:!!isRetry};
  ctrl=new AbortController();
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal:ctrl.signal});
    if(!res.ok){const j=await res.json().catch(()=>({}));const er=new Error(j.error||('HTTP '+res.status));er.nice=!!j.error;if(j.limit)setTok(j);throw er}
    const rd=res.body.getReader(),dec=new TextDecoder();let buf='';
    for(;;){
      const {value,done}=await rd.read();if(done)break;
      buf+=dec.decode(value,{stream:true});let k;
      while((k=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,k).trim();buf=buf.slice(k+1);if(!line)continue;
        const ev=JSON.parse(line);
        if(ev.t==='status'){if(!answer)b.stat.textContent=ev.v}
        else if(ev.t==='text'){answer=ev.v;b.stat.textContent='';b.out.innerHTML=md(ev.v)}
        else if(ev.t==='tool'){b.steps.insertAdjacentHTML('beforeend',stepHTML(ev));b.out.innerHTML='';answer=''}
        else if(ev.t==='done'){conv=conv.concat([{role:'user',text}],ev.messages);nNew=1+ev.messages.length;finalText=(ev.messages[ev.messages.length-1]||{}).text||answer;const s=ev.stats||{};if(s.limit)setTok(s);
          b.foot.textContent=s.tokens+' tokens \u00b7 '+s.tps+' tok/s \u00b7 context '+(s.ctx||0)+'/'+INFO.seq+((s.ctx||0)>INFO.seq*0.7?' \u2014 getting long, tap New chat for best answers':'')}
        else if(ev.t==='error'){b.out.innerHTML='<span class="errt">'+esc(ev.v)+'</span>'}
        msgs.scrollTop=msgs.scrollHeight;
      }
    }
  }catch(e){
    if(e.name!=='AbortError')b.out.innerHTML='<span class="errt">'+(e.nice?'':'Could not reach the server: ')+esc(e.message)+'</span>';
    else if(answer){conv=conv.concat([{role:'user',text},{role:'assistant',text:answer}]);nNew=2;finalText=answer}
  }
  b.dots.remove();b.stat.textContent='';busy=false;ctrl=null;setBtn(false);addActions(b,text,nNew,finalText);msgs.scrollTop=msgs.scrollHeight;
}
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.isComposing){e.preventDefault();send()}});
fetch('/api/info').then(r=>r.json()).then(i=>{INFO=i;if(!INFO.models.find(m=>m.id===S.model&&m.ok))S.model=(INFO.models.find(m=>m.ok)||{id:'chat'}).id;applyUI()});
fetch('/api/usage').then(r=>r.json()).then(setTok).catch(()=>{});
</script>
</body>
</html>'''

LOGIN_HTML = r'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bee Admin login</title>
<style>*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:radial-gradient(circle at 20% 20%,#f5f3ff,#e6e9f2);font:15px system-ui,"Segoe UI",sans-serif;color:#18181b}
form{background:#fff;padding:28px;border-radius:22px;width:min(360px,92vw);box-shadow:0 20px 50px -15px rgba(60,40,140,.25)}h1{margin:0 0 4px;font-size:22px}p{margin:0 0 16px;color:#71717a;font-size:13px}
input{width:100%;padding:12px 14px;margin:6px 0;border:1px solid #e4e4e7;border-radius:12px;font:inherit}button{width:100%;margin-top:10px;padding:12px;border:0;border-radius:12px;background:#5e5ce6;color:#fff;font:inherit;font-weight:700;cursor:pointer}.e{color:#d33;font-size:13px;min-height:18px}</style></head>
<body><form method="post"><h1>&#128029; Bee Admin</h1><p>Sign in to see what people ask and what Bee answers.</p>
<input name="user" placeholder="Username" autocomplete="username" required autofocus><input name="password" type="password" placeholder="Password" autocomplete="current-password" required>
<div class="e">__MSG__</div><button>Sign in</button></form></body></html>'''

ADMIN_HTML = r'''<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bee Admin</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f2f2f7;color:#18181b;font:14px/1.5 system-ui,"Segoe UI",Roboto,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:14px 22px;background:#fff;border-bottom:1px solid #e4e4e7;position:sticky;top:0;z-index:5}header b{font-size:17px}header a{color:#5e5ce6;text-decoration:none;font-weight:600}
main{max-width:1200px;margin:0 auto;padding:18px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.card{background:#fff;border-radius:16px;padding:14px 16px;box-shadow:0 4px 18px -8px rgba(60,40,140,.18)}.card small{color:#71717a;display:block}.card b{font-size:26px;letter-spacing:-.02em}
.box{background:#fff;border-radius:18px;padding:16px;margin-top:14px;box-shadow:0 4px 18px -8px rgba(60,40,140,.18)}h2{margin:0 0 10px;font-size:16px;display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
.tabs button,.pg button{border:1px solid #e4e4e7;background:#fff;border-radius:999px;padding:4px 14px;cursor:pointer;font:inherit}.tabs .on{background:#5e5ce6;color:#fff;border-color:#5e5ce6}
.chart{display:flex;align-items:flex-end;gap:5px;height:170px;overflow-x:auto;padding-top:18px}.col{flex:1;min-width:26px;height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;font-size:11px;color:#71717a}
.bar{width:100%;background:linear-gradient(#8b7cf6,#5e5ce6);border-radius:6px 6px 0 0;min-height:2px}table{width:100%;border-collapse:collapse}th{text-align:left;color:#71717a;font-weight:600;font-size:12px;padding:6px 8px}
td{padding:8px;border-top:1px solid #eee;vertical-align:top}.cell{max-height:3em;overflow:hidden;white-space:pre-wrap;word-break:break-word}tr.open .cell{max-height:none}tr.row{cursor:pointer}tr.row:hover{background:#faf9ff}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:11px;background:#ebe8ff;color:#4340c9;margin:0 3px 3px 0}.tag.r{background:#fff1d6;color:#9a5b00}.tag.e{background:#ffe1e1;color:#b42318}.tag.s{background:#eee;color:#555}
input[type=search]{padding:7px 12px;border:1px solid #e4e4e7;border-radius:999px;font:inherit;min-width:220px}.pg{display:flex;gap:8px;align-items:center;margin-top:10px;justify-content:flex-end;color:#71717a}.w{overflow-x:auto}</style></head>
<body><header><b>&#128029; Bee Admin</b><span><span id="upd" style="color:#71717a;margin-right:14px"></span><a href="/logout">Log out</a></span></header>
<main><div class="cards" id="cards"></div>
<div class="box"><h2><span>Requests</span><span class="tabs"><button data-p="day" class="on">Daily</button> <button data-p="week">Weekly</button> <button data-p="month">Monthly</button></span></h2><div class="chart" id="chart"></div><div class="w"><table id="ptab"></table></div></div>
<div class="box"><h2><span>Questions &amp; answers <small id="tot" style="color:#71717a;font-weight:400"></small></span><input type="search" id="q" placeholder="Search question, answer or visitor"></h2>
<div class="w"><table><thead><tr><th>Time</th><th>Visitor</th><th>Question</th><th>Answer</th><th>Tokens</th><th></th></tr></thead><tbody id="rows"></tbody></table></div>
<div class="pg"><button id="prev">&larr; Newer</button><span id="pgn"></span><button id="next">Older &rarr;</button></div></div></main>
<script>const $=id=>document.getElementById(id),esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let per='day',page=0,q='',ST=null,TOT=0;
async function api(u){const r=await fetch(u);if(r.status===401){location='/login';return null}return r.json()}
function drawStats(){const s=ST;if(!s)return;
 $('cards').innerHTML=[['Requests today',s.today],['This week',s.week],['This month',s.month],['All time',s.total],['Tokens today',s.tokens_today],['Retries today',s.retries_today],['Visitors today',s.visitors_today],['Visitors total',s.visitors_total]].map(x=>'<div class="card"><small>'+x[0]+'</small><b>'+x[1]+'</b></div>').join('');
 const rows=s['by_'+per],mx=Math.max(1,...rows.map(r=>r.n));
 $('chart').innerHTML=rows.map(r=>'<div class="col" title="'+r.p+': '+r.n+' requests, '+(r.t||0)+' tokens, '+r.u+' visitors, '+(r.r||0)+' retries"><span>'+r.n+'</span><div class="bar" style="height:'+(r.n/mx*85)+'%"></div><span>'+esc(per==='day'?r.p.slice(5):r.p)+'</span></div>').join('')||'<p>No data yet</p>';
 $('ptab').innerHTML='<tr><th>Period</th><th>Requests</th><th>Retries</th><th>Tokens</th><th>Visitors</th></tr>'+rows.slice().reverse().map(r=>'<tr><td>'+esc(r.p)+'</td><td>'+r.n+'</td><td>'+(r.r||0)+'</td><td>'+(r.t||0)+'</td><td>'+r.u+'</td></tr>').join('')}
async function loadStats(){ST=await api('/api/admin/stats');drawStats();$('upd').textContent='Updated '+new Date().toLocaleTimeString()}
async function loadLogs(){const d=await api('/api/admin/logs?page='+page+'&q='+encodeURIComponent(q));if(!d)return;TOT=d.total;
 $('tot').textContent='('+d.total+')';$('pgn').textContent=d.total?('Page '+(page+1)+' of '+Math.ceil(d.total/25)):'No results';$('prev').disabled=page===0;$('next').disabled=(page+1)*25>=d.total;
 $('rows').innerHTML=d.rows.map(r=>'<tr class="row" onclick="this.classList.toggle(\'open\')"><td>'+new Date(r.ts*1000).toLocaleString()+'</td><td><code>'+esc((r.sid||'').slice(0,6))+'</code></td><td style="min-width:200px"><div class="cell">'+esc(r.question)+'</div></td><td style="min-width:260px"><div class="cell">'+esc(r.answer)+'</div></td><td>'+r.tokens+'</td><td>'
  +(r.retry?'<span class="tag r">retry</span>':'')+(r.status==='error'?'<span class="tag e">error</span>':'')+(r.status==='stopped'?'<span class="tag s">stopped</span>':'')+'<span class="tag s">'+esc(r.model)+'</span>'+(r.tools?'<span class="tag">'+esc(r.tools)+'</span>':'')+'</td></tr>').join('')}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{per=b.dataset.p;document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));drawStats()});
$('prev').onclick=()=>{page--;loadLogs()};$('next').onclick=()=>{page++;loadLogs()};
let tm;$('q').oninput=e=>{clearTimeout(tm);tm=setTimeout(()=>{q=e.target.value;page=0;loadLogs()},300)};
loadStats();loadLogs();setInterval(()=>{loadStats();if(page===0&&!q)loadLogs()},20000);</script></body></html>'''

if __name__ == '__main__':
    main()
