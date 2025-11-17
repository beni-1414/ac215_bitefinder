/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://orchestrator:9000/:path*', // proxy to orchestrator inside docker
      },
    ]
  },
}

module.exports = nextConfig
