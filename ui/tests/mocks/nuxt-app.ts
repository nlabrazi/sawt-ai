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

export function useRequestURL() {
  return new URL('http://localhost:3000/')
}

export function useHead(_input: unknown) {}
