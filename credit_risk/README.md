# Credit Risk Model

## Train
```
python credit_risk/train_credit_model.py
```
Artifacts saved to `credit_risk/artifacts`.

## Run API
```
uvicorn credit_risk.api:app --reload --port 8000
```
POST JSON example:
```json
{
  "data": {
    "DerogCnt": 1,
    "CollectCnt": 0,
    "BanruptcyInd": 0,
    "InqCnt06": 3,
    "InqTimeLast": 2,
    "InqFinanceCnt24": 1,
    "TLTimeFirst": 120,
    "TLTimeLast": 10,
    "TLCnt03": 0,
    "TLCnt12": 2,
    "TLCnt24": 5,
    "TLCnt": 10,
    "TLSum": 15000,
    "TLMaxSum": 20000,
    "TLSatCnt": 12,
    "TLDel60Cnt": 0,
    "TLBadCnt24": 0,
    "TL75UtilCnt": 1,
    "TL50UtilCnt": 2,
    "TLBalHCPct": 60,
    "TLSatPct": 50,
    "TLDel3060Cnt24": 0,
    "TLDel90Cnt24": 0,
    "TLDel60CntAll": 0,
    "TLOpenPct": 55,
    "TLBadDerogCnt": 0,
    "TLDel60Cnt24": 0,
    "TLOpen24Pct": 45
  }
}
```
