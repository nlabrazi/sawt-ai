# ROLE
# ----
# Container de dev Nuxt pour Sawt AI.

FROM node:22

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# The persistent node_modules volume can outlive the image and its lockfile.
CMD ["sh", "-c", "npm ci && exec npm run dev -- --host 0.0.0.0"]
