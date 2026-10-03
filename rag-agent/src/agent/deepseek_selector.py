"""Optional official API selector. Credentials never enter state/events/errors."""
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .actions import InvalidAction
from .selector import selector_messages, parse_selector_action


class ApiCostBudget:
    """Conservative peak-rate reservation; retain bounds on unmetered requests."""
    def __init__(self, *, input_rmb_per_million, output_rmb_per_million, stop_rmb=3.0):
        values=(input_rmb_per_million,output_rmb_per_million,stop_rmb)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in values) or stop_rmb>5:
            raise ValueError("invalid API cost budget")
        self.input_rate, self.output_rate, self.stop_rmb = values
        self.reserved_rmb = 0.0

    def reserve(self, input_bytes, max_tokens):
        # UTF-8 byte count plus generous protocol overhead bounds token input.
        amount=((input_bytes+4096)*self.input_rate+max_tokens*self.output_rate)/1e6
        if self.reserved_rmb+amount>=self.stop_rmb:
            raise RuntimeError("deepseek:cost_limit")
        self.reserved_rmb+=amount
        return amount

    def settle(self, reservation, usage):
        if all(k in usage for k in ('prompt_tokens','completion_tokens')):
            actual=(usage['prompt_tokens']*self.input_rate+usage['completion_tokens']*self.output_rate)/1e6
            self.reserved_rmb+=actual-reservation


class DeepSeekActionSelector:
    def __init__(self, *, timeout=120, max_tokens=512, retries=1, budget=None,
                 action_contract="copied_quote"):
        if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<=0:
            raise ValueError("invalid selector timeout")
        if type(max_tokens) is not int or not 1<=max_tokens<=512 or type(retries) is not int or retries not in (0,1):
            raise ValueError("invalid selector limits")
        if action_contract not in ("copied_quote", "evidence_reference"):
            raise ValueError("invalid action contract")
        self.timeout, self.max_tokens, self.retries = timeout, max_tokens, retries
        self.action_contract = action_contract
        self.budget=budget
        self.events=[]

    def __call__(self, state):
        key=os.environ.get('DEEPSEEK_API_KEY','').strip()
        if not key:
            raise RuntimeError('deepseek:missing_key')
        payload=dict(model='deepseek-flash', messages=selector_messages(state,contract=self.action_contract),
                     thinking={'type':'disabled'}, response_format={'type':'json_object'},
                     stream=False, temperature=0, max_tokens=self.max_tokens)
        data=json.dumps(payload,ensure_ascii=False).encode('utf-8')
        for attempt in range(self.retries+1):
            reservation=self.budget.reserve(len(data),self.max_tokens) if self.budget else 0
            event=dict(retry=bool(attempt),usage={},error=None)
            self.events.append(event)
            request=Request('https://api.deepseek.com/chat/completions',data=data,
                            headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
            try:
                with urlopen(request,timeout=self.timeout) as response:
                    raw=response.read(128001)
                if len(raw)>128000:
                    raise InvalidAction('response too large')
                body=json.loads(raw)
                usage=body.get('usage') or {}
                event['usage']={k:v for k,v in usage.items() if k in (
                    'prompt_tokens','completion_tokens','total_tokens',
                    'prompt_cache_hit_tokens','prompt_cache_miss_tokens') and type(v) is int and v>=0}
                if self.budget: self.budget.settle(reservation,event['usage'])
                content=body['choices'][0]['message']['content']
                if not isinstance(content,str) or not content.strip():
                    raise InvalidAction('empty action content')
                if body['choices'][0].get('finish_reason')=='length':
                    raise InvalidAction('truncated action')
                return parse_selector_action(content,state,contract=self.action_contract)
            except HTTPError as exc:
                event['error']='http_'+str(exc.code)
                exc.close()
                if exc.code in (429,500,502,503,504) and attempt<self.retries:
                    continue
                raise RuntimeError('deepseek:'+event['error']) from None
            except (TimeoutError,URLError) as exc:
                timeout=isinstance(exc,TimeoutError) or isinstance(getattr(exc,'reason',None),TimeoutError)
                event['error']='timeout' if timeout else 'network_error'
                if attempt<self.retries: continue
                if timeout: raise TimeoutError('deepseek:timeout') from None
                raise RuntimeError('deepseek:network_error') from None
            except InvalidAction:
                event['error']='invalid_action'
                raise
            except (ValueError,TypeError,KeyError,IndexError,AttributeError):
                event['error']='malformed_json'
                raise InvalidAction('deepseek:malformed_json') from None
