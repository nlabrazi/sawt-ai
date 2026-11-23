# syntax=docker/dockerfile:1
ARG NODE_IMAGE=node:26-bookworm-slim

##################################
# 1) BASE (deps + source)        #
##################################
FROM ${NODE_IMAGE} AS base
WORKDIR /app

COPY package*.json .npmrc ./
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc,required=true npm ci

COPY . .

##################################
# 2) DEV (hot-reload Nuxt)       #
##################################
FROM base AS dev
ENV NODE_ENV=development
EXPOSE 3000
# Only reinstall at startup if a mounted node_modules volume is empty.
CMD ["sh", "-c", "[ -x node_modules/.bin/nuxt ] || npm ci; exec npm run dev -- --host 0.0.0.0"]

##################################
# 3) BUILD (Nuxt production)     #
##################################
FROM base AS build
ENV NODE_ENV=production
RUN npm run build

##################################
# 4) PROD (runtime minimal)      #
##################################
FROM ${NODE_IMAGE} AS prod
WORKDIR /app
ENV NODE_ENV=production
ENV HOST=0.0.0.0
ENV PORT=3000
ENV NITRO_HOST=0.0.0.0
ENV NITRO_PORT=3000

COPY --from=build /app/.output ./.output
COPY --from=build /app/package*.json ./

EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
