// Adapté du chargeur hCaptcha du portfolio de Nabil Labrazi.
type CaptchaOptions = {
  sitekey: string
  theme: string
  size: string
  hl: string
  callback: (token: string) => void
  'expired-callback': () => void
  'error-callback': () => void
}

export type CaptchaApi = {
  render: (element: HTMLElement, options: CaptchaOptions) => string
  reset: (id: string) => void
  remove: (id: string) => void
}

declare global {
  interface Window {
    hcaptcha?: CaptchaApi
    sawtCaptchaReady?: () => void
  }
}

let pending: Promise<CaptchaApi> | undefined

export function loadCaptcha(): Promise<CaptchaApi> {
  if (window.hcaptcha) return Promise.resolve(window.hcaptcha)
  if (pending) return pending

  pending = new Promise<CaptchaApi>((resolve, reject) => {
    const script = document.createElement('script')
    const timeout = window.setTimeout(fail, 15_000)
    function fail() {
      window.clearTimeout(timeout)
      script.remove()
      pending = undefined
      delete window.sawtCaptchaReady
      reject(new Error('Captcha unavailable'))
    }
    window.sawtCaptchaReady = () => {
      if (!window.hcaptcha) return fail()
      window.clearTimeout(timeout)
      delete window.sawtCaptchaReady
      resolve(window.hcaptcha)
    }
    script.src =
      'https://js.hcaptcha.com/1/api.js?render=explicit&recaptchacompat=off&onload=sawtCaptchaReady'
    script.async = true
    script.onerror = fail
    document.head.append(script)
  })
  return pending
}
