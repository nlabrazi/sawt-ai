# ROLE
# ----
# Container de dev Nuxt pour Sawt AI.

FROM node:20

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

CMD ["sh", "-c", "if [ ! -x node_modules/.bin/nuxt ]; then npm install; fi && npm run dev -- --host 0.0.0.0"]
