# tpl_befe_docker_gae

Took the time to set this up one time so that i could bootstrap a new dockerized set of services (backend and frontend) easier.  It is preconfigured to be deployed to a GCP / Google App Engine. There are some files that are not included in this repo that need to be set up before it can be used.

# Backend
* app.yaml for deploying to GCP
* /secrets/.env (will need "OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"
* /secrets/gcp-credential.json (and the file name should be updated in the .env)

#Frontend
* app.yaml for deploying to GCP
* edit Dockerfile to update local backend url:port
