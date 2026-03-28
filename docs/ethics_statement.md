# Ethics and Policy Statement

## Intended Use
This system is an educational forecasting and scheduling prototype. It is designed for offline analysis and classroom demonstration.

## Key Risks
1. **User autonomy risk**
   Automated scheduling may recommend delays that users do not want.
   - Mitigation: document override and opt-out as a required deployment safeguard.

2. **Equity risk**
   A model trained on one household may not generalize to households with different routines, appliances, or socioeconomic conditions.
   - Mitigation: clearly state limited representativeness and avoid broad claims.

3. **Transparency risk**
   Users may mistake predicted demand and proxy costs for exact real-world savings.
   - Mitigation: label all outputs as simulation-based and provide baseline comparisons.

## Privacy and Consent
The chosen dataset contains no personal identifiers. Still, future deployment should minimize data collection, anonymize household data, and require informed consent.

## Fairness
The project does not support population-level fairness claims because the dataset covers only one household. This limitation must be disclosed.

## Misuse Considerations
The system should not be used to force energy restrictions on users without transparency, consent, and manual override.
