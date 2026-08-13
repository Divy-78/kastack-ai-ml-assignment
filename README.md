# KaStack Message Intelligence

Local, security-first NLP system for the KaStack AI/ML Intern assignment.

## Six required categories
- Action Required
- Meeting or Event
- Personal Information
- General Information
- Promotional
- Sensitive Information

## Important data rule
The supplied 900-message dataset and separate mandatory-ID file are NOT included in this public package. The Streamlit app accepts both files through uploaders.

## Architecture
Uploaded CSVs -> chronological sorting -> sensitive-data shield -> high-precision local rules -> local TF-IDF word+character Logistic Regression fallback -> six-category classification -> deterministic task/event extraction -> structured outputs.

## Classification
The supplied dataset has no answer labels. Therefore this project does not claim supervised ground-truth accuracy. It uses transparent weak supervision:
1. Sort messages chronologically.
2. Apply high-precision local rules for sensitive, promotional, event, action and personal-information cues.
3. Use those rule outputs as weak labels for a local TF-IDF word+character Logistic Regression model.
4. Let sensitive rules take precedence over the statistical model.
5. Use the local model as a fallback.
6. Store a confidence signal and a short reason.

`message_id`, row number and sender are not used as model features, avoiding positional/sender shortcuts.

Confidence is a rule/model confidence signal, not a validated accuracy score or calibrated probability because answer labels were not supplied.

## Task/event extraction
The local deterministic extractor returns title, description, date/deadline, time, explicit person, priority and source message ID. Missing information remains `null` or `unresolved`. `tomorrow` is resolved only from the message timestamp; other ambiguous relative dates remain unresolved.

## Sensitive-data shield
Local regex/rules detect passwords, OTP/PIN, card numbers, bank accounts, recovery codes, access/authentication tokens, identification numbers, private addresses/contact details, login-detail notices and health-test-result values. Detected values are replaced with typed placeholders before display. High-risk credentials/financial/identity/health data use conservative actions such as `do_not_store` or `do_not_send_external`.

Raw messages are not sent to external AI services.

## Run
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
Then upload `messages.csv` and `mandatory_demo_ids.csv`.

## Outputs
The dashboard downloads classification, mandatory-15, task/event and sensitive-result CSVs. Do not commit raw assignment data or unmasked sensitive values.

## Limitations
- No answer labels were supplied; weak supervision is explicitly documented.
- Rules are strongest for patterns represented in this synthetic assignment dataset.
- Fallback is a lightweight local model, not an LLM.
- Person extraction is conservative.
- Confidence is not ground-truth calibrated.
- Unseen language can be uncertain.

## AI-tool usage declaration

AI tools were used during development for brainstorming, architecture discussion, debugging assistance, code review, and documentation support. The implementation was reviewed and tested locally, and the final system uses custom logic and open-source Python libraries for runtime processing. Raw assignment messages were not sent to external AI services.

## Public GitHub checklist
Do NOT commit:
- messages.csv
- mandatory_demo_ids.csv
- raw screenshots containing sensitive-looking values
- logs containing unmasked sensitive values

## Loom checklist
Show the running system and visibly cover:
1. overview/system flow
2. dataset structure without sensitive values
3. all six categories
4. all 15 mandatory IDs
5. 3+ tasks
6. 3+ meetings/events
7. one missing/unclear field
8. sensitive detection, masking, risk and action
9. three classification decisions with explanations
10. one uncertain/incorrect result and why
11. one important code section explained in your own words
12. limitations and improvements
