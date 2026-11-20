# AI Use Disclosure

**Course:** DSA 8401 Applied Machine Learning, Capstone Project
**Student:** Jimcollins Wamae Wanguku
**Tool used:** Microsoft Copilot (chat assistant), November 2026

This file discloses how AI assistance was used in this project, as required by the course policy. I take responsibility for every design decision, line of code and result in this repository.

## 1. Where AI assistance was used

| Area | How AI helped | My role |
|---|---|---|
| Project set-up | Suggested a folder structure, a `Makefile`, the pipeline stages and tests | Chose the problem, the stocks and the time split; ran and checked every step on my machine |
| Environment and tooling | Step-by-step guidance on Python 3.12, virtual environments, VS Code, Git/GitHub and Docker on macOS | Installed, configured and fixed issues locally |
| Notebook (EDA and modelling) | Drafted notebook code for loading, cleaning, EDA charts, feature engineering, EGARCH, XGBoost, the hybrid model, evaluation and Diebold-Mariano tests | Supplied the data, ran it end to end, reported errors, reviewed outputs and interpreted the results |
| Debugging | Helped diagnose errors (Python version conflicts, a missing `scikit-learn` dependency, missing hybrid columns from a fragile merge) and rewrote the hybrid section so it no longer depends on cell order | Identified the failures and confirmed the fixes |
| Forecast app | Drafted the Streamlit app, its Dockerfile and requirements | Specified the design (a drop-down, at most a few controls, a dotted forecast line and a disclaimer) and requested the 1-month and 3-month horizons |
| Model safeguards | Proposed capping volatility and fading the return signal after I found unrealistic forecasts (KQ at KES 5.86 forecast to KES 873 in 5 days) | Spotted the problem, tested the fix across stocks and accepted it |
| Documentation | Drafted this README, the AI use disclosure and the slide deck from my notebook outputs | Reviewed, corrected and approved all content; all numbers come from my own notebook run |
| Visuals | Generated an illustrative image of a stock exchange trading floor for the slides | The image is illustrative only. It is not real NSE data, and any prices shown in it are not real |

## 2. What AI was not used for
- **Data:** AI did not create, alter or invent any data. All results come from the NSE data I collected, processed by code I ran.
- **Results:** AI did not choose or adjust the reported numbers. Every metric in the README and slides is copied from my notebook outputs.
- **Problem selection and judgement:** choosing the problem, the stocks, the time split, the honest reporting of weak results (such as the KQ EGARCH instability and the hybrid's non-significant gain over XGBoost) and the final conclusions were my decisions.

## 3. How AI output was checked
- Ran all code on my own machine against the full dataset (`Restart → Run All` for the notebook; `make pipeline` and `make test` for the pipeline).
- Checked results for leakage (time-based split, EGARCH fitted on training data only) and sanity-checked forecasts across all 10 stocks and every horizon.
- Rejected or asked for fixes to AI output that failed or gave unrealistic results.
- Confirmed I can explain every part of the code and every modelling choice.

## 4. Limitations of AI assistance
- Some AI-suggested code had bugs (for example, a typo in a fix and an unstable multi-step EGARCH simulation). These were found through testing and fixed.
- AI cannot verify the original data source. Data provenance is my responsibility.
