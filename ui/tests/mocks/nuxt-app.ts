type RuntimeConfig = {
  public: {
    apiBaseUrl: string
    contactEmail?: string
    web3formsAccessKey?: string
    siteUrl?: string
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

let requestUrl = 'http://localhost:3000/'

export function useRequestURL() {
  return new URL(requestUrl)
}

export function setRequestURL(url: string) {
  requestUrl = url
}

export function useHead(_input: unknown) {}
