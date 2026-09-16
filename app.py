"""Flask UI and prediction API. Loads only the local, trusted training artifact."""
import io
import json
import math
from pathlib import Path
import pickle
import sys
sys.dont_write_bytecode = True
if __name__ == "__main__":
    sys.modules["app"] = sys.modules[__name__]
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request, send_file
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
"""Deterministic features shared by training and Flask inference."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

CATEGORICAL = ['Gender', 'City', 'Income_Level', 'Contract_Type', 'Plan_Type',
               'Payment_Method', 'Upsell_Downgrade_History']
NUMERIC = ['Age', 'Tenure_Months', 'Auto_Renewal_Flag', 'Usage_Frequency',
           'Days_Since_Last_Activity', 'Feature_Usage_Count', 'Avg_Session_Duration_Min',
           'Monthly_Charges', 'Discount_Applied_Pct', 'Late_Payment_Count',
           'Support_Tickets_Raised', 'Complaint_Count', 'Avg_Resolution_Time_Hrs',
           'Satisfaction_Score', 'Referral_Count', 'Loyalty_Program_Member', 'Email_Open_Rate_Pct']
RAW_FEATURES = NUMERIC + CATEGORICAL
ENGINEERED = ['Has_Complaint', 'Has_Late_Payment', 'Complaints_Per_Tenure_Month',
              'Late_Payments_Per_Tenure_Month', 'Tickets_Per_Tenure_Month',
              'Estimated_Charge_After_Discount', 'Inactivity_x_NonLoyalty',
              'Complaints_x_LowSatisfaction']


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        out = X.loc[:, RAW_FEATURES].copy()
        for c in NUMERIC:
            out[c] = pd.to_numeric(out[c], errors='coerce').astype(float)
        for c in CATEGORICAL:
            out[c] = out[c].map(lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else np.nan).astype(object)
        tenure = out.Tenure_Months.where(out.Tenure_Months > 0)
        out['Has_Complaint'] = out.Complaint_Count.gt(0).astype(float).where(out.Complaint_Count.notna())
        out['Has_Late_Payment'] = out.Late_Payment_Count.gt(0).astype(float).where(out.Late_Payment_Count.notna())
        out['Complaints_Per_Tenure_Month'] = out.Complaint_Count / tenure
        out['Late_Payments_Per_Tenure_Month'] = out.Late_Payment_Count / tenure
        out['Tickets_Per_Tenure_Month'] = out.Support_Tickets_Raised / tenure
        out['Estimated_Charge_After_Discount'] = out.Monthly_Charges * (1 - out.Discount_Applied_Pct / 100)
        out['Inactivity_x_NonLoyalty'] = out.Days_Since_Last_Activity * (1-out.Loyalty_Program_Member)
        out['Complaints_x_LowSatisfaction'] = out.Complaint_Count * (10-out.Satisfaction_Score)
        return out.replace([np.inf, -np.inf], np.nan)


ROOT=Path(__file__).resolve().parent

PAGE_HTML = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Staywell · Customer care</title><link rel="stylesheet" href="/static/style.css"></head>\n<body><header class="site-header"><a class="brand" href="/" aria-label="Staywell home"><span class="brand-icon">s</span>staywell<span class="brand-dot">.</span></a><a class="header-link" href="#how">How it works <span>↗</span></a></header>\n<main><section class="hero"><div><p class="eyebrow"><span></span> A LITTLE ATTENTION GOES A LONG WAY</p><h1>Give customers<br>a reason to <em>stay.</em></h1><p class="intro">Spot who may need a little extra care, and make your next conversation count.</p><a class="button primary" href="#check">Check a customer <span>↗</span></a><div class="hero-note"><span>01</span> Understand <i>→</i><span>02</span> Reach out <i>→</i><span>03</span> Reconnect</div></div><div class="hero-art" aria-hidden="true"><div class="orbit orbit-one"></div><div class="orbit orbit-two"></div><div class="art-note"><span class="spark">✦</span> Small gestures.<br><strong>Stronger connections.</strong></div><div class="connection-card"><span class="avatar">A</span><div><strong>A thoughtful check-in</strong><small>can make all the difference.</small></div><span class="heart">♡</span></div><div class="art-label">BUILT AROUND PEOPLE</div></div></section>\n<section class="how" id="how"><div><span class="step-number">01</span><h3>Share what you know</h3><p>Add details about a customer\'s experience. Leave anything you don\'t know blank.</p></div><div><span class="step-number">02</span><h3>Get a helpful indication</h3><p>See whether this customer may need extra attention.</p></div><div><span class="step-number">03</span><h3>Make a personal connection</h3><p>Use the result to start a conversation and understand their needs.</p></div></section>\n<section id="check" class="workspace"><div class="section-title"><div><p class="eyebrow">CUSTOMER CHECK-IN</p><h2>Start with their story.</h2><p>No names or contact details needed.</p></div><button class="text-button" id="example" type="button">Try an example ↗</button></div><div class="workspace-grid"><form id="customer-form"><div class="form-heading"><span>Customer details</span><button type="button" id="reset" class="text-button">Clear all</button></div><p class="form-help">More complete details can make the result more useful. Unknown details will be filled with typical values.</p>\n{% for title,description,fields in groups %}<details class="field-group" {% if loop.first %}open{% endif %}><summary><span class="group-number">0{{ loop.index }}</span><span><strong>{{ title }}</strong><small>{{ description }}</small></span><span class="expand">+</span></summary><div class="fields">{% for name,label,hint in fields %}{% set spec=schema[name] %}<label for="{{ name }}">{{ label }}{% if name in [\'Auto_Renewal_Flag\',\'Loyalty_Program_Member\'] %}<select id="{{ name }}" name="{{ name }}"><option value="">Not sure</option><option value="1">Yes</option><option value="0">No</option></select>{% elif spec.type == \'category\' %}<select id="{{ name }}" name="{{ name }}"><option value="">Not sure</option>{% for option in spec.choices %}<option value="{{ option }}">{{ option }}</option>{% endfor %}</select>{% else %}<input id="{{ name }}" name="{{ name }}" type="number" placeholder="Not sure" min="{{ spec.min }}" {% if spec.max is not none %}max="{{ spec.max }}"{% endif %} step="{{ \'1\' if spec.integer else \'any\' }}">{% endif %}<small>{{ hint }}</small></label>{% endfor %}</div></details>{% endfor %}<p id="form-status" role="status"></p><button class="button primary submit" type="submit">See customer outlook <span>→</span></button></form>\n<aside class="result-card" id="result" aria-live="polite" aria-atomic="true"><div class="result-top"><span class="eyebrow">YOUR CUSTOMER OUTLOOK</span><span class="status-dot"></span></div><div id="result-icon" class="result-icon" aria-hidden="true">♡</div><h2 id="result-title">A little insight.<br>A better conversation.</h2><p id="result-description">Add customer details, then select “See customer outlook” to get started.</p><div id="next-step" hidden><span class="eyebrow">A GOOD NEXT STEP</span><p id="advice"></p></div><div class="result-note">This is a guide, not a certainty. It cannot tell you when someone will leave. Always check in with the customer before making decisions.</div></aside></div></section>\n<section class="closing"><span>♡</span><h2>Retention starts with listening.</h2><p>The most useful next step is often a simple “How can we help?”</p></section></main><footer><a class="brand" href="/">staywell.</a><span>Better conversations. Lasting connections.</span></footer><script id="example-data" type="application/json">{{ defaults|tojson }}</script><script src="/static/app.js"></script></body></html>'
STYLES = "@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');\n:root{--ink:#24342d;--muted:#68766d;--paper:#faf9f6;--green:#315c47;--line:#e2e6de;--lime:#e4edb6}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:25px}body{margin:0;background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif;font-size:15px;line-height:1.6}button,input,select{font:inherit}button,a,input,select,summary{-webkit-tap-highlight-color:transparent}a{color:inherit;text-decoration:none}button,summary{cursor:pointer}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #93ae72;outline-offset:5px}.site-header,main,footer{max-width:1200px;margin:auto}.site-header{display:flex;align-items:center;justify-content:space-between;padding:26px 34px;border-bottom:1px solid var(--line)}.brand{display:flex;gap:8px;align-items:center;font:800 27px 'Manrope',sans-serif;letter-spacing:-1.6px}.brand-icon{background:var(--green);color:white;width:32px;height:34px;display:grid;place-items:center;border-radius:10px 10px 4px 10px;font-size:27px;font-style:italic}.brand-dot{color:#83a264;margin-left:-8px}.header-link{font-size:13px}.header-link span{margin-left:18px}main{padding:0 34px}.hero{display:grid;grid-template-columns:1.25fr 1fr;gap:45px;padding:76px 0 62px;align-items:center}.eyebrow{font-size:10px;letter-spacing:1.8px;font-weight:700}.hero .eyebrow{display:flex;align-items:center;gap:9px}.eyebrow>span{height:7px;width:7px;border-radius:50%;background:#87a968}h1,h2,h3,p{margin:0}h1{font-family:'Manrope',sans-serif;font-weight:500;font-size:clamp(42px,5vw,66px);line-height:1.12;letter-spacing:-3.5px;margin:22px 0}h1 em{font-family:Georgia,serif;color:var(--green);font-weight:400}p.intro{max-width:390px;color:var(--muted);font-size:17px;line-height:1.7}.button{display:inline-flex;align-items:center;justify-content:space-between;gap:40px;padding:15px 23px;border-radius:8px;font-size:14px;font-weight:600;border:0}.primary{background:var(--green);color:white}.primary:hover{background:#234735}.hero .button{margin-top:29px}.hero-note{display:flex;gap:9px;align-items:center;margin-top:30px;font-size:11px;color:var(--muted)}.hero-note span{font-size:9px;border:1px solid #d8dfd2;border-radius:50%;padding:3px 5px}.hero-note i{font-style:normal;margin:0 4px;color:#b5bfb3}.hero-art{height:370px;background:#edf0e5;border-radius:150px 150px 18px 18px;position:relative;overflow:hidden;display:flex;flex-direction:column;justify-content:center;align-items:center}.orbit{border:1px solid #d4dec6;border-radius:50%;position:absolute;width:340px;height:340px;top:75px;left:-75px}.orbit-two{left:125px;top:-135px;width:390px;height:390px}.art-note{text-align:center;font-size:24px;line-height:1.45;z-index:1;letter-spacing:-.7px}.art-note strong{font-weight:500}.spark{display:block;color:#789655;font-size:40px;margin-bottom:14px}.connection-card{z-index:1;display:flex;align-items:center;gap:13px;padding:17px 20px;background:#fffdf8;border:1px solid white;box-shadow:0 12px 35px #3f543310;border-radius:12px;margin-top:28px;transform:rotate(-4deg);font-size:12px}.avatar{background:#eedccc;color:#765235;width:39px;height:39px;border-radius:50%;display:grid;place-items:center;font-family:Georgia,serif;font-size:22px}.connection-card small{display:block;color:var(--muted);font-size:11px}.heart{font-size:26px;margin-left:10px;color:var(--green)}.art-label{font-size:8px;letter-spacing:2px;margin-top:28px;color:#75836b}.how{display:grid;grid-template-columns:repeat(3,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:29px 0;gap:40px}.how>div{padding-left:31px;position:relative}.step-number{position:absolute;left:0;top:3px;color:#92a080;font-size:11px}.how h3{font-size:14px;font-weight:600;margin-bottom:6px}.how p{font-size:12px;line-height:1.8;color:var(--muted)}.workspace{padding-top:63px}.section-title{display:flex;justify-content:space-between;align-items:center;margin-bottom:26px}.section-title h2{font:500 32px 'Manrope',sans-serif;letter-spacing:-1.2px;margin:8px 0}.section-title p:not(.eyebrow){color:var(--muted);font-size:13px}.text-button{border:0;background:transparent;color:var(--green);font-size:12px;font-weight:600;padding:7px 0;text-decoration:underline;text-underline-offset:4px}.workspace-grid{display:grid;grid-template-columns:1.65fr 1fr;gap:28px;align-items:start}form{border:1px solid var(--line);background:#fff;border-radius:14px;padding:27px}.form-heading{display:flex;justify-content:space-between;align-items:center;font-weight:600;font-size:16px}.form-help{font-size:12px;color:var(--muted);margin:10px 0 22px;max-width:450px}.field-group{border-top:1px solid var(--line)}summary{list-style:none;display:flex;align-items:center;gap:13px;padding:20px 0}summary::-webkit-details-marker{display:none}.group-number{font-size:10px;background:#f0f3eb;border-radius:7px;padding:6px 8px;color:#748165}summary strong{font-size:13px;font-weight:600;display:block}summary small{font-size:11px;color:var(--muted)}.expand{margin-left:auto;font-size:22px;font-weight:400;color:#829078}details[open] .expand{transform:rotate(45deg)}.fields{display:grid;grid-template-columns:1fr 1fr;gap:18px;padding:0 0 25px}label{font-size:12px;font-weight:500}label small{display:block;font-size:10px;font-weight:400;color:var(--muted);margin-top:5px}input,select{display:block;width:100%;min-width:0;background:#fcfcfa;color:var(--ink);border:1px solid #dbe1d6;border-radius:7px;padding:10px 11px;margin-top:7px;height:43px;font-size:13px}input:focus,select:focus{outline:2px solid #adc292;outline-offset:1px}input::placeholder{color:#90998c}.submit{width:100%;margin-top:20px}button:disabled{opacity:.65;cursor:wait}#form-status{font-size:12px;color:#71502e;margin-top:12px}.result-card{background:#edf1e7;border:1px solid #e0e7d8;border-radius:14px;padding:29px;position:sticky;top:25px}.result-top{display:flex;justify-content:space-between;align-items:center}.result-top .eyebrow{font-size:9px}.status-dot{width:6px;height:6px;border-radius:50%;background:#90a378}.result-icon{width:63px;height:63px;border-radius:50%;background:#dde7cf;display:grid;place-items:center;font-size:33px;color:#637e4d;margin:39px 0 23px}.result-card h2{font:500 27px/1.3 'Manrope',sans-serif;letter-spacing:-1px}.result-card>p{font-size:13px;color:#66745e;line-height:1.9;margin-top:17px}.result-note{border-top:1px solid #d7dfce;margin-top:32px;padding-top:20px;font-size:11px;line-height:1.8;color:#707b66}#next-step{background:#ffffff80;padding:17px;border-radius:9px;margin-top:23px}#advice{font-size:13px;margin-top:7px}.result-card.has-result #result-title{font-size:40px;font-weight:800;line-height:1.15;color:#205b36;background:#d9edda;border-left:5px solid #34844b;padding:18px;border-radius:9px;letter-spacing:-1px}.result-card.has-result.attention #result-title{color:#854410;background:#ffe1ae;border-left-color:#b66a17}.result-card.attention{background:#f6eedf;border-color:#eadbc1}.attention .result-icon{background:#eedbbb;color:#92682e}.closing{text-align:center;padding:70px 10px 62px}.closing>span{color:#769563;font-size:30px}.closing h2{font:500 26px 'Manrope',sans-serif;letter-spacing:-.7px;margin:12px 0}.closing p{font-size:13px;color:var(--muted)}footer{border-top:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;padding:25px 34px 35px}footer .brand{font-size:20px}footer>span{font-size:11px;color:var(--muted)}[hidden]{display:none!important}@media(max-width:850px){.hero{gap:25px}.hero-art{height:330px}.workspace-grid{grid-template-columns:1.4fr 1fr}.result-card{padding:22px}h1{letter-spacing:-2px}.connection-card{padding:12px;gap:8px}.how{gap:18px}}@media(max-width:650px){.site-header{padding:20px}.brand{font-size:24px}main{padding:0 20px}.hero{grid-template-columns:1fr;padding:40px 0}.hero-art{height:285px;max-width:420px;width:100%;margin:auto}.hero h1{font-size:46px}.hero .eyebrow{font-size:8px}.hero-note{gap:7px}.how{grid-template-columns:1fr;gap:22px}.how p{font-size:13px}.workspace{padding-top:40px}.workspace-grid{grid-template-columns:1fr}.section-title{gap:15px;align-items:flex-end}.section-title h2{font-size:28px}.section-title>.text-button{max-width:100px}.result-card{position:static}.result-icon{margin:24px 0 18px}form{padding:20px}.fields{gap:16px 12px}.closing{padding:45px 0}footer{padding:20px;gap:20px}footer>span{text-align:right;max-width:180px}.intro{font-size:16px!important}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}"
BROWSER_SCRIPT = "const form=document.querySelector('#customer-form'), card=document.querySelector('#result'), title=document.querySelector('#result-title'), description=document.querySelector('#result-description'), status=document.querySelector('#form-status'), next=document.querySelector('#next-step');\nfunction resetResult(){card.classList.remove('attention','has-result');title.textContent='A little insight. A better conversation.';description.textContent='Add customer details, then select “See customer outlook” to get started.';next.hidden=true;status.textContent='';document.querySelector('#result-icon').textContent='♡';}\ndocument.querySelector('#reset').addEventListener('click',()=>{form.reset();resetResult();});\ndocument.querySelector('#example').addEventListener('click',()=>{const values=JSON.parse(document.querySelector('#example-data').textContent);Object.entries(values).forEach(([k,v])=>{form.elements[k].value=v;});resetResult();status.textContent='Example details added. These do not belong to a real customer.';});\nform.addEventListener('input',()=>{resetResult();});form.addEventListener('invalid',event=>{const group=event.target.closest('details');if(group)group.open=true;},true);\nform.addEventListener('submit',async event=>{event.preventDefault();const body={};new FormData(form).forEach((v,k)=>body[k]=v===''?null:v);if(Object.values(body).every(v=>v===null)){status.textContent='Please add at least one detail, or try the example above.';form.querySelector('input').focus();return;}const button=form.querySelector('[type=submit]');button.disabled=true;card.setAttribute('aria-busy','true');card.classList.remove('has-result','attention');title.textContent='Taking a closer look…';description.textContent='Your customer outlook will be ready in a moment.';next.hidden=true;try{const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!response.ok)throw new Error('Please check your details and try again.');const data=await response.json(), p=data.predictions[0], attention=p.prediction===1;card.classList.toggle('attention',attention);title.textContent=attention?'May leave':'Likely to stay';card.classList.add('has-result');description.textContent=attention?'These details suggest this customer may be considering leaving. A thoughtful conversation could help you understand why.':'These details suggest this customer may stay. Keep listening and make it easy for them to get help.';document.querySelector('#result-icon').textContent=attention?'↗':'♡';document.querySelector('#advice').textContent=attention?'Ask how their experience has been, listen to any concerns, and agree on one practical way to help.':'Thank them for choosing you, ask what is working well, and let them know you are available.';next.hidden=false;status.textContent=p.imputed_fields.length?'Some details were left blank. Typical values were used for those details.':'';if(window.innerWidth<651)card.scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth',block:'start'});}catch(error){title.textContent='Let’s try that again';description.textContent='We could not complete the check. Review the details and try again in a moment.';status.textContent='Unable to complete your request. Please try again.';}finally{button.disabled=false;card.removeAttribute('aria-busy');}});"
FORM_GROUPS = [('Their relationship with you', 'A little background', [('Age', 'Age', 'In years'), ('Gender', 'Gender', 'Choose if known'), ('City', 'City', 'Where they are based'), ('Income_Level', 'Income level', 'Choose if known'), ('Tenure_Months', 'Time as a customer', 'Number of months'), ('Loyalty_Program_Member', 'Loyalty member', 'Are they in your loyalty program?')]), ('Their plan & payments', 'Subscriptions and billing', [('Contract_Type', 'Subscription type', 'Current agreement'), ('Plan_Type', 'Plan', 'Current service plan'), ('Payment_Method', 'Payment method', 'How they usually pay'), ('Monthly_Charges', 'Monthly bill', 'Use the same currency as your records'), ('Discount_Applied_Pct', 'Discount (%)', 'From 0 to 100'), ('Auto_Renewal_Flag', 'Automatic renewal', 'Does the plan renew automatically?'), ('Late_Payment_Count', 'Late payments', 'Number of late payments'), ('Upsell_Downgrade_History', 'Plan changes', 'Any upgrades or downgrades?')]), ('How they use your service', 'Activity and engagement', [('Usage_Frequency', 'Number of visits', 'Use the period in your customer records'), ('Days_Since_Last_Activity', 'Days since last visit', 'Days since they last used the service'), ('Feature_Usage_Count', 'Features used', 'Number of different features used'), ('Avg_Session_Duration_Min', 'Typical visit length', 'In minutes'), ('Email_Open_Rate_Pct', 'Emails opened (%)', 'Share of emails they open'), ('Referral_Count', 'People referred', 'Number of referrals')]), ('Their experience', 'Feedback and support', [('Support_Tickets_Raised', 'Help requests', 'Number of times they contacted support'), ('Complaint_Count', 'Complaints', 'Number of complaints raised'), ('Avg_Resolution_Time_Hrs', 'Typical time to resolve an issue', 'In hours'), ('Satisfaction_Score', 'Satisfaction rating', 'Use the rating in your customer records')])]


def create_app(model_path=None):
    # Load only the model.pkl supplied with this application.
    with open(model_path or ROOT/'model.pkl', 'rb') as stream:
        bundle=pickle.load(stream)
    metadata=bundle['metadata']
    schema=metadata['schema']
    app=Flask(__name__, static_folder=None)
    app.config.update(MAX_CONTENT_LENGTH=2*1024*1024, JSON_SORT_KEYS=False)
    app.json.sort_keys=False

    def validate(records):
        if not isinstance(records,list) or not 1<=len(records)<=1000:
            raise ValueError('Supply between 1 and 1,000 customer records.')
        clean=[];warnings=[]
        for i,record in enumerate(records,1):
            if not isinstance(record,dict):raise ValueError(f'Row {i}: customer must be an object.')
            missing=[c for c in RAW_FEATURES if c not in record]
            if missing:raise ValueError(f'Row {i}: missing columns: {", ".join(missing)}. Use null for an unknown value.')
            extra=set(record)-set(RAW_FEATURES)-{'Customer_ID','Signup_Date','Total_Charges','Churn_Flag'}
            if extra:raise ValueError(f'Row {i}: unknown columns: {", ".join(sorted(extra))}.')
            row={};unknown=[];imputed=[]
            for col,spec in schema.items():
                val=record[col]
                if val is None or (isinstance(val,str) and not val.strip()):
                    row[col]=np.nan;imputed.append(col);continue
                if spec['type']=='category':
                    if not isinstance(val,str):raise ValueError(f'Row {i}: {col} must be text.')
                    row[col]=val.strip()
                    if row[col] not in spec['choices']:unknown.append(col)
                else:
                    if isinstance(val,(bool,list,dict)):raise ValueError(f'Row {i}: {col} must be numeric.')
                    try:number=float(val)
                    except (ValueError,TypeError,OverflowError):raise ValueError(f'Row {i}: {col} must be numeric.')
                    if not math.isfinite(number):raise ValueError(f'Row {i}: {col} must be finite or null.')
                    if number<spec['min'] or (spec['max'] is not None and number>spec['max']):
                        raise ValueError(f'Row {i}: {col} is outside the allowed range.')
                    if spec['integer'] and not number.is_integer():raise ValueError(f'Row {i}: {col} must be a whole number.')
                    row[col]=number
            if len(imputed)==len(schema):raise ValueError(f'Row {i}: at least one feature must be supplied.')
            clean.append(row)
            warnings.append({'imputed_fields':imputed,'unseen_categories':unknown})
        return pd.DataFrame(clean,columns=RAW_FEATURES),warnings

    def predict(records):
        frame,warnings=validate(records)
        scores=bundle['pipeline'].predict_proba(frame)[:,1]
        return [{'row':i+1,'churn_score':round(float(p),6),
                 'prediction':int(p>=bundle['threshold']),
                 'label':'Likely churn' if p>=bundle['threshold'] else 'Likely retained',
                 **warnings[i]} for i,p in enumerate(scores)]

    @app.get('/')
    def index():
        return render_template_string(PAGE_HTML, schema=schema, groups=FORM_GROUPS, defaults={name: spec["default"] for name, spec in schema.items()})

    @app.get('/static/<filename>', endpoint='static')
    def static_asset(filename):
        if filename == 'style.css':
            return app.response_class(STYLES, mimetype='text/css')
        if filename == 'app.js':
            return app.response_class(BROWSER_SCRIPT, mimetype='text/javascript')
        return jsonify(error='Asset not found.'),404

    @app.get('/health')
    def health():
        return jsonify(status='ok',model=bundle['model_name'],threshold=bundle['threshold'])

    @app.get('/api/schema')
    def api_schema():
        return jsonify(schema=schema,required_columns=RAW_FEATURES,maximum_batch_rows=1000)

    @app.get('/api/metrics')
    def api_metrics():
        return jsonify(selected_model=metadata['selected_model'],results=metadata['results'],splits=metadata['splits'])

    @app.post('/api/predict')
    def api_predict():
        if not request.is_json:return jsonify(error='Content-Type must be application/json.'),415
        payload=request.get_json()
        if isinstance(payload,dict) and 'customers' in payload:records=payload['customers']
        elif isinstance(payload,dict):records=[payload]
        else:records=payload
        try:predictions=predict(records)
        except ValueError as exc:return jsonify(error=str(exc)),400
        return jsonify(model=bundle['model_name'],threshold=bundle['threshold'],predictions=predictions,
                       note='Uncalibrated model scores; educational demo, feature timing unverified.')

    @app.post('/api/predict-csv')
    def api_predict_csv():
        upload=request.files.get('file')
        if upload is None:return jsonify(error='Choose a CSV file.'),400
        try:
            # Text input prevents pandas from silently accepting infinity or dropping unknown strings.
            frame=pd.read_csv(upload,dtype=str,keep_default_na=False,nrows=1001)
            if frame.empty:raise ValueError('CSV must contain customer rows.')
            predictions=predict(frame.to_dict(orient='records'))
        except (ValueError,pd.errors.ParserError,UnicodeDecodeError) as exc:
            return jsonify(error=str(exc)),400
        result=pd.DataFrame([{k:v for k,v in p.items() if k not in ['imputed_fields','unseen_categories']} |
                             {'imputed_fields':'; '.join(p['imputed_fields']),'unseen_categories':'; '.join(p['unseen_categories'])}
                             for p in predictions])
        # Output has row numbers rather than user-supplied IDs, avoiding spreadsheet formula injection.
        return send_file(io.BytesIO(result.to_csv(index=False).encode('utf-8')),mimetype='text/csv',as_attachment=True,download_name='churn_predictions.csv')

    @app.get('/sample.csv')
    def sample_csv():
        sample=pd.DataFrame([{c:s['default'] for c,s in schema.items()}]).to_csv(index=False)
        return send_file(io.BytesIO(sample.encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='sample_batch.csv')

    @app.get('/reports/<name>')
    def report_file(name):
        allowed={'eda.png','distributions.png','correlations.png','evaluation_curves.png','confusion_matrices.png','feature_importance.png','evaluation_report.md','model_metrics.csv'}
        if name not in allowed:return jsonify(error='Report not found.'),404
        mime = 'image/png' if name.endswith('.png') else 'text/csv' if name.endswith('.csv') else 'text/markdown'
        return send_file(io.BytesIO(bundle['reports'][name]), mimetype=mime, download_name=name)

    @app.errorhandler(BadRequest)
    def bad_request(exc):return jsonify(error='Malformed request or invalid JSON.'),400

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(exc):return jsonify(error='Upload exceeds the 2 MB limit.'),413

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        return response

    return app


if __name__=='__main__':
    from waitress import serve
    application=create_app()
    print('Customer Churn app running at http://127.0.0.1:5000', flush=True)
    print('Press Ctrl+C to stop.', flush=True)
    serve(application,host='127.0.0.1',port=5000,threads=4)
