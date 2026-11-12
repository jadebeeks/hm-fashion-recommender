# Architecture, scalability and monitoring

Owner: Member 5 (with input from all).

- **Batch vs real time:** e.g. retrain nightly, store top-12 per customer, serve from a cache.
- **Cold-start routing:** where the segment switch happens.
- **Scaling:** what changes going from our subset to all customers.
- **Monitoring:** what we watch after launch (hit rate, coverage, drift in popular items, errors).
- **Retraining schedule and fallbacks.**
