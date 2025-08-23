"use client";
import { useState } from 'react';
import axios from 'axios';

const initialState = {
  age: '',
  annual_income: '',
  employment_years: '',
  derogatory_marks: '',
  inquiries_last6m: '',
  inquiries_finance_24m: '',
  total_accounts: '',
  active_accounts: '',
  high_credit_util_75: '',
  util_50_plus: '',
  balance_high_credit_pct: '',
  satisfied_pct: '',
  delinquency_30_60_24m: '',
  delinquency_90d_24m: '',
  delinquencies_60d: '',
  chargeoffs_last24m: '',
  derog_or_bad_cnt: '',
  accounts_open_last24m: '',
  max_account_balance: '',
  total_balance: ''
};

export default function Home() {
  const [form, setForm] = useState(initialState);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null); setResult(null);
    // Prepare payload: convert numeric strings
    const payload = {};
    Object.keys(form).forEach(k => {
      if (form[k] !== '') {
        const v = parseFloat(form[k]);
        if (!isNaN(v)) payload[k] = v; else payload[k] = form[k];
      }
    });
    try {
      const res = await axios.post('http://127.0.0.1:8000/predict', payload, { headers: { 'Content-Type': 'application/json' } });
      setResult(res.data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => { setForm(initialState); setResult(null); };

  return (
    <main style={{ maxWidth: 900, margin: '30px auto', fontFamily: 'system-ui' }}>
      <h1>Credit Risk Predictor</h1>
      <p>Enter applicant financial & credit attributes. Leave blank to use defaults.</p>
      <form onSubmit={handleSubmit} style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(200px,1fr))', gap: 12 }}>
        {Object.keys(initialState).map(field => (
          <div key={field} style={{ display: 'flex', flexDirection: 'column' }}>
            <label style={{ fontSize: 12, fontWeight: 600 }}>{field}</label>
            <input name={field} value={form[field]} onChange={handleChange} placeholder={field} style={{ padding: 6, border: '1px solid #ccc', borderRadius: 4 }} />
          </div>
        ))}
        <div style={{ gridColumn: '1 / -1', display: 'flex', gap: 12 }}>
          <button type="submit" disabled={loading} style={{ padding: '10px 18px' }}>{loading ? 'Predicting...' : 'Predict'}</button>
          <button type="button" onClick={resetForm}>Reset</button>
        </div>
      </form>

      {error && <div style={{ marginTop: 20, color: 'red' }}>Error: {error}</div>}
      {result && (
        <div style={{ marginTop: 30, padding: 20, border: '1px solid #ddd', borderRadius: 8 }}>
          <h2>Result</h2>
          <p>Status: <strong>{result.status}</strong></p>
          <p>Prediction: {result.prediction}</p>
          <p>Probability Bad: {(result.probability_bad * 100).toFixed(2)}%</p>
          <p>Probability Good: {(result.probability_good * 100).toFixed(2)}%</p>
          <p>Threshold Used: {result.threshold_used}</p>
          <p>Model Version: {result.model_version}</p>
        </div>
      )}
    </main>
  );
}
