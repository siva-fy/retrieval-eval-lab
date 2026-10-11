# Event replay

An idempotency key identifies the same event across retries. Store the key and the resulting state change in the same transaction to avoid applying a payment twice.

A conflicting payload with the same idempotency key must be rejected for investigation. Duplicate delivery is expected; contradictory delivery is a data integrity problem.
