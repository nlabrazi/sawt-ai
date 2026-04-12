type RuntimeConfig = {
  public: {
    apiBaseUrl: string
  }
}

let runtimeConfig: RuntimeConfig = {
  public: {
    apiBaseUrl: 'http://localhost:8000',
  },
}

export function useRuntimeConfig() {
  return runtimeConfig
}

export function setRuntimeConfig(nextConfig: RuntimeConfig) {
  runtimeConfig = nextConfig
}
