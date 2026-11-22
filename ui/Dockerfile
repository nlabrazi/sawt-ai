# syntax=docker/dockerfile:1
# ROLE
# ----
# Container de dev Nuxt pour Sawt AI.

FROM node:26

WORKDIR /app

COPY package*.json .npmrc ./
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc,required=true npm ci

COPY . .

# Only reinstall at startup if a mounted node_modules volume is empty.
CMD ["sh", "-c", "[ -x node_modules/.bin/nuxt ] || npm ci; exec npm run dev -- --host 0.0.0.0"]
