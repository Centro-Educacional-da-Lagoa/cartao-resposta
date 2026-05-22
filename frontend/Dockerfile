FROM node:22-alpine AS build
WORKDIR /app

ARG VITE_DATA_API_URL=http://localhost:3001
ARG VITE_BOT_API_URL=http://localhost:5000
ARG VITE_GOOGLE_CLIENT_ID=
ENV VITE_DATA_API_URL=$VITE_DATA_API_URL
ENV VITE_BOT_API_URL=$VITE_BOT_API_URL
ENV VITE_GOOGLE_CLIENT_ID=$VITE_GOOGLE_CLIENT_ID

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine AS runner
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
