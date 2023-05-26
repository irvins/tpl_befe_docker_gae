# tpl_befe_docker_gae

Took the time to set this up one time so that i could bootstrap a new dockerized set of services (backend and frontend) easier.  It is preconfigured to be deployed to a GCP / Google App Engine. There are some files that are not included in this repo that need to be set up before it can be used.

# Backend
* app.yaml for deploying to GCP
* /secrets/.env (will need "OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"
* /secrets/gcp-credential.json (and the file name should be updated in the .env)

# Frontend
* app.yaml for deploying to GCP
* edit Dockerfile to update local backend url:port

Once the proper files are included in a terminal window from the root of the project (where the docker-compose.yml) is, simply do a:

\>\> docker compose build (if it is the first time and any subsequent time where new packages are added or Dockerfile changed)

then

\>\> docker compose up

once satisfied, to deploy to Google Cloud App Engine:

first confirm that you are signed in to the right project by using 

\>\> gcloud config list

use 

\>\> gcloud projects list

\>\> gcloud config set project [PROJECT_ID] if necessary

once confirmed, to deploy simply type

\>\> ./deploy.sh

and wait
