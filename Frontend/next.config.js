/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
  },

  output: 'standalone',

  // Important: disable assetPrefix to fix loading of _next static files over IP/HTTPS
  assetPrefix: '',
};

module.exports = nextConfig;

