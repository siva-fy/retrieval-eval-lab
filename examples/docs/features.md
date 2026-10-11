# Historical features

Point-in-time joins only use events that happened before the prediction timestamp and were available to the system by that time. This prevents leakage from late arrivals.

A feature window has explicit boundaries. The lower boundary can be inclusive while the prediction timestamp is exclusive. Keep the convention identical during training and serving.
