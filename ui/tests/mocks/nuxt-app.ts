type RuntimeConfig = {
  public: {
    apiBaseUrl: string
    contactEmail?: string
  }
}

let runtimeConfig: RuntimeConfig = {
  public: {
    apiBaseUrl: 'http://localhost:8000',
    contactEmail: '',
  },
}

export function useRuntimeConfig() {
  return runtimeConfig
}

export function setRuntimeConfig(nextConfig: RuntimeConfig) {
  runtimeConfig = nextConfig
}
