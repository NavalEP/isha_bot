/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  images: { unoptimized: true },
  env: {
    CAREPAY_API_URL: 'http://localhost:8000/api/v1/agent',
    NEXT_PUBLIC_CAREPAY_API_URL: 'http://localhost:8000/api/v1/agent',
  },
};

module.exports = nextConfig;
