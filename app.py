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

PAGE_HTML = '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Customer Churn · ML Project</title><link rel="stylesheet" href="{{ url_for(\'static\',filename=\'style.css\') }}"></head>\n<body><header><div class="brand"><span class="mark">C</span> Customer Intelligence</div><span class="pill">Flask · Local demo</span></header>\n<main><section class="intro"><div><p class="eyebrow">CUSTOMER CHURN / MACHINE LEARNING</p><h1>Understand retention.<br>Explore churn predictions.</h1><p class="lede">Compare three models, inspect their results, and score customer profiles using one consistent prediction pipeline.</p></div><aside><span class="eyebrow">SELECTED ON VALIDATION DATA</span><h2>{{ metadata.selected_model }}</h2><p>Decision threshold <strong>{{ \'%.3f\'|format(metadata.threshold) }}</strong></p><p class="muted">Model scores are uncalibrated. This is an educational demo; feature timing and the churn horizon are unverified.</p></aside></section>\n<section class="stats"><article><span>Customers</span><strong>{{ \'{:,}\'.format(metadata.quality.rows) }}</strong></article><article><span>Overall churn</span><strong>{{ \'%.1f\'|format(metadata.quality.target_counts[\'1\']/metadata.quality.rows*100) }}%</strong></article><article><span>Train / Validation / Test · 70/15/15</span><strong class="split">{{ metadata.splits.train.rows }} / {{ metadata.splits.validation.rows }} / {{ metadata.splits.test.rows }}</strong></article><article><span>Selected model · Test accuracy</span><strong>{{ \'%.1f\'|format(metadata.results[metadata.selected_model].test.accuracy*100) }}%</strong></article></section>\n<nav aria-label="Project sections"><button class="tab active" data-panel="predict">Predict churn</button><button class="tab" data-panel="evaluate">Model evaluation</button><button class="tab" data-panel="explore">EDA & features</button></nav>\n<section id="predict" class="panel"><div class="section-head"><div><p class="eyebrow">01 / TRY THE MODEL</p><h2>Customer profile</h2><p>Starting values are training medians and modes, not a real customer. Blank fields use training-set imputation.</p></div><button type="button" id="reset" class="secondary">Reset example</button></div>\n<div class="predict-layout"><form id="customer-form"><div class="fields">{% for name,spec in schema.items() %}<label>{{ name.replace(\'_\',\' \') }}{% if spec.type == \'category\' %}<select name="{{ name }}"><option value="">Unknown</option>{% for option in spec.choices %}<option value="{{ option }}" {% if option == spec.default %}selected{% endif %}>{{ option }}</option>{% endfor %}</select>{% else %}<input name="{{ name }}" type="number" value="{{ spec.default }}" min="{{ spec.min }}" {% if spec.max is not none %}max="{{ spec.max }}"{% endif %} step="{{ \'1\' if spec.integer else \'any\' }}">{% endif %}</label>{% endfor %}</div><button class="primary" type="submit">Predict churn <span aria-hidden="true">→</span></button></form>\n<aside class="result" aria-live="polite"><p class="eyebrow">PREDICTION</p><h2 id="prediction-label">Ready when you are</h2><div id="score">—</div><p id="result-detail">Enter a customer profile and run a prediction.</p><div class="track"><div id="score-bar"></div></div><p class="muted">The label uses the validation-selected threshold. A score is not a verified probability of future churn.</p></aside></div>\n<div class="batch"><div><h2>Batch predictions</h2><p>Upload up to 1,000 rows (2 MB). Download predictions in the same row order.</p><a href="/sample.csv">Download the CSV template</a></div><form id="batch-form"><label class="sr-only" for="csv">Customer CSV</label><input id="csv" type="file" name="file" accept=".csv" required><button type="submit" class="secondary">Score CSV</button><p id="batch-status" role="status"></p></form></div></section>\n<section id="evaluate" class="panel" hidden><div class="section-head"><div><p class="eyebrow">02 / HELD-OUT TEST RESULTS</p><h2>Model comparison</h2><p>Models and thresholds were selected before test evaluation. Each row uses that model’s validation-selected threshold.</p></div><a href="/reports/model_metrics.csv" class="secondary">Download metrics</a></div><div class="table-scroll"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>AP</th></tr></thead><tbody>{% for name,result in metadata.results.items() %}<tr {% if name == metadata.selected_model %}class="selected"{% endif %}><th>{{ name }}{% if name == metadata.selected_model %}<small>Selected on validation</small>{% endif %}</th>{% for key in [\'accuracy\',\'precision\',\'recall\',\'f1\',\'roc_auc\',\'pr_auc\',\'average_precision\'] %}<td>{{ \'%.3f\'|format(result.test[key]) }}</td>{% endfor %}</tr>{% endfor %}</tbody></table></div><p class="muted">PR-AUC uses trapezoidal area. Average precision (AP) uses recall-weighted precision and is reported separately. Model and threshold selection maximize validation accuracy; review recall alongside accuracy.</p><img src="/reports/evaluation_curves.png" alt="ROC and precision-recall curves for all models" loading="lazy"><img src="/reports/confusion_matrices.png" alt="Test confusion matrices: actual classes in rows and predicted classes in columns" loading="lazy"><a href="/reports/evaluation_report.md">Download the full evaluation report</a></section>\n<section id="explore" class="panel" hidden><div class="section-head"><div><p class="eyebrow">03 / DATA AND FEATURES</p><h2>Explore the training data</h2><p>Detailed charts use training rows. Earlier full-dataset exploration means this test is not a fully blind prospective evaluation.</p></div></div><div class="notes"><article><h3>Data quality</h3><p>{{ metadata.quality.missing_cells }} missing cells · {{ metadata.quality.duplicate_customer_ids }} duplicate IDs.</p><p>{{ metadata.quality.inactivity_exceeds_days_since_signup }} records have inactivity longer than the time since signup as of {{ metadata.quality.audit_date }}. The snapshot date needs confirmation.</p></article><article><h3>Preprocessing</h3><p>Training-only median/mode imputation, numeric scaling, and one-hot category encoding. Customer ID, signup date, and total charges are excluded.</p></article><article><h3>Engineered features</h3><p>Complaint and payment flags, counts per tenure month, estimated discounted charges, and complaint/satisfaction and inactivity/loyalty interactions.</p></article></div><img src="/reports/eda.png" alt="Training churn distribution and segment comparisons" loading="lazy"><img src="/reports/distributions.png" alt="Numeric feature distributions by churn" loading="lazy"><img src="/reports/feature_importance.png" alt="Validation permutation feature importance" loading="lazy"><details><summary>Numeric correlations</summary><img src="/reports/correlations.png" alt="Training numeric correlation matrix" loading="lazy"></details></section>\n<footer>Customer Churn ML Project · Logistic Regression / Random Forest / XGBoost · Flask</footer></main><script src="{{ url_for(\'static\',filename=\'app.js\') }}"></script></body></html>\n'
STYLES = ':root{font-family:Segoe UI,Arial,sans-serif;color:#1e343c;background:#f4f6f5;font-synthesis:none}*{box-sizing:border-box}body{margin:0}header{height:76px;background:#fff;border-bottom:1px solid #dce3e1;display:flex;align-items:center;justify-content:space-between;padding:0 5%}.brand{display:flex;gap:12px;align-items:center;font-weight:650}.mark{background:#194e51;color:white;padding:7px 12px;border-radius:9px;font-size:22px}.pill{background:#edf3f1;color:#416563;border-radius:30px;padding:8px 15px;font-size:12px}main{max-width:1260px;margin:auto;padding:44px 30px}.intro{display:grid;grid-template-columns:1.7fr 1fr;gap:56px;align-items:center}.eyebrow{font-size:11px;letter-spacing:1.8px;font-weight:700;color:#487579;margin:0 0 12px}h1{font-size:43px;line-height:1.15;font-weight:620;letter-spacing:-1.6px;margin:0 0 18px}h2{font-weight:620;font-size:23px;margin:0 0 12px}h3{font-size:16px}p{line-height:1.6;font-size:14px;color:#5a6c71}.lede{font-size:16px;max-width:560px}.intro aside{border-left:3px solid #c8d9d4;padding:16px 24px}.muted{font-size:12px;color:#697b7e}.stats{display:grid;grid-template-columns:1fr 1fr 1.5fr 1fr;gap:16px;margin:32px 0}.stats article{padding:22px;background:white;border:1px solid #e0e7e4;border-radius:12px}.stats span{display:block;font-size:12px;color:#627577;margin-bottom:10px}.stats strong{font-size:30px;font-weight:600}.stats .split{font-size:25px}nav{display:flex;gap:24px;border-bottom:1px solid #cddbd7;margin-bottom:28px}.tab{border:0;background:transparent;border-bottom:3px solid transparent;border-radius:0;padding:14px 0;color:#64787a;font-weight:600}.tab.active{color:#1a5558;border-bottom-color:#1a5558}button{font:inherit;cursor:pointer;border-radius:7px;padding:11px 18px;font-size:13px}button:disabled{opacity:.55;cursor:wait}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #72b5bb;outline-offset:3px}.panel{background:#fff;border:1px solid #dee7e3;border-radius:14px;padding:28px}.section-head{display:flex;justify-content:space-between;align-items:start;gap:20px;margin-bottom:24px}.section-head p{margin-bottom:0}.secondary{background:#fff;border:1px solid #cbdad7;text-decoration:none;color:#285d60;white-space:nowrap;display:inline-block;border-radius:7px;padding:10px 15px;font-size:13px}.primary{background:#1c585b;color:white;border:0;min-width:180px;margin-top:26px}.primary span{margin-left:20px}.predict-layout{display:grid;grid-template-columns:2.3fr 1fr;gap:28px}.fields{display:grid;grid-template-columns:repeat(3,1fr);gap:19px 17px}label{font-size:12px;color:#486267;display:block}input,select{display:block;width:100%;padding:10px 11px;border:1px solid #cddbd7;border-radius:6px;margin-top:7px;font:inherit;color:#1f363d;background:white;font-size:14px;min-height:41px}.result{background:#f0f5f3;border:1px solid #dce7e2;border-radius:10px;padding:24px;align-self:start;position:sticky;top:20px}.result h2{font-size:20px}#score{font-size:54px;color:#205d60;font-weight:600;margin:20px 0}.track{height:7px;background:#d9e4df;border-radius:9px;margin:23px 0;overflow:hidden}#score-bar{height:100%;width:0;background:#297b7b;transition:width .35s}.batch{border-top:1px solid #e0e7e4;display:grid;grid-template-columns:1fr 1fr;gap:30px;margin-top:32px;padding-top:28px}.batch h2{font-size:20px}.batch form{display:flex;align-items:start;flex-wrap:wrap;gap:12px}.batch input{max-width:280px;margin:0}.batch p[role]{width:100%;margin:0}.batch a,a{color:#256b70}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:15px 12px;border-bottom:1px solid #dfe8e4;text-align:right;white-space:nowrap}th:first-child{text-align:left}thead{background:#f3f6f5;color:#50686b}tr.selected{background:#eaf4ee}small{display:block;color:#38756d;font-size:10px;margin-top:4px;font-weight:400}img{display:block;width:100%;height:auto;border:1px solid #edf1ef;border-radius:8px;margin:25px 0}.notes{display:grid;grid-template-columns:repeat(3,1fr);gap:22px}.notes article{padding:18px;background:#f3f6f4;border-radius:9px}footer{text-align:center;font-size:11px;color:#748886;margin:35px 0 0}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}details{padding:14px;background:#f5f7f6}summary{cursor:pointer;font-size:14px}[hidden]{display:none!important}@media(max-width:900px){h1{font-size:34px}.intro{gap:20px}.fields{grid-template-columns:repeat(2,1fr)}.stats{grid-template-columns:repeat(2,1fr)}.notes{grid-template-columns:1fr}.predict-layout{grid-template-columns:1.8fr 1fr}}@media(max-width:650px){main{padding:25px 14px}.intro,.predict-layout,.batch{grid-template-columns:1fr}.intro aside{padding:10px 20px}.panel{padding:18px}.result{position:static}.fields{grid-template-columns:1fr 1fr}.stats strong,.stats .split{font-size:22px}.stats article{padding:16px}nav{gap:18px}.tab{font-size:12px}.section-head{flex-wrap:wrap}.pill{display:none}h1{font-size:32px}}\n'
BROWSER_SCRIPT = "document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b===button));document.querySelectorAll('.panel').forEach(p=>p.hidden=p.id!==button.dataset.panel);}));\nconst form=document.querySelector('#customer-form');\ndocument.querySelector('#reset').addEventListener('click',()=>{form.reset();document.querySelector('#prediction-label').textContent='Ready when you are';document.querySelector('#score').textContent='—';document.querySelector('#result-detail').textContent='Enter a customer profile and run a prediction.';document.querySelector('#score-bar').style.width='0';});\nform.addEventListener('submit',async event=>{event.preventDefault();const button=form.querySelector('button[type=submit]');button.disabled=true;document.querySelector('#prediction-label').textContent='Calculating…';document.querySelector('#score').textContent='—';document.querySelector('#score-bar').style.width='0';try{const body={};new FormData(form).forEach((v,k)=>body[k]=v===''?null:v);const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();if(!response.ok)throw Error(data.error||'Prediction failed.');const p=data.predictions[0];document.querySelector('#prediction-label').textContent=p.label;document.querySelector('#score').textContent=(p.churn_score*100).toFixed(1)+'%';document.querySelector('#score-bar').style.width=(p.churn_score*100)+'%';document.querySelector('#result-detail').textContent='Model score · threshold '+data.threshold.toFixed(3)+(p.imputed_fields.length?' · '+p.imputed_fields.length+' unknown fields imputed.':'')+(p.unseen_categories.length?' · Unseen categories: '+p.unseen_categories.join(', '):'');}catch(error){document.querySelector('#prediction-label').textContent='Check your input';document.querySelector('#result-detail').textContent=error.message;}finally{button.disabled=false;}});\ndocument.querySelector('#batch-form').addEventListener('submit',async event=>{event.preventDefault();const button=event.target.querySelector('button');const status=document.querySelector('#batch-status');button.disabled=true;status.textContent='Scoring…';try{const response=await fetch('/api/predict-csv',{method:'POST',body:new FormData(event.target)});if(!response.ok){const data=await response.json();throw Error(data.error||'Upload failed.');}const url=URL.createObjectURL(await response.blob());const a=document.createElement('a');a.href=url;a.download='churn_predictions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status.textContent='Predictions downloaded.';}catch(error){status.textContent=error.message;}finally{button.disabled=false;}});\n"


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
        return render_template_string(PAGE_HTML, metadata=metadata, schema=schema)

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
