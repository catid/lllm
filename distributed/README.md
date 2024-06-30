# Asynchronous Training

This is a simple implementation of Async DiLoCo: https://arxiv.org/pdf/2401.09135v1

The central orchestrator is responsible for:
* Hosting an HTTPS server for communications on a public address.
* Maintaining the dataset and reference model outputs for loss targets.
