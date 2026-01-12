import type { NextConfig } from "next";

const basePath = process.env.BASE_PATH || "";
const assetPrefix = process.env.ASSET_PREFIX || basePath || undefined;

const nextConfig: NextConfig = {
  output: "export",
  basePath: basePath || undefined,
  assetPrefix,
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
