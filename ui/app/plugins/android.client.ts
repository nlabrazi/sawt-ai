import { defineNuxtPlugin } from '#app'
import { Capacitor, SystemBars } from '@capacitor/core'
import { SplashScreen } from '@capacitor/splash-screen'

export default defineNuxtPlugin((nuxtApp) => {
  if (Capacitor.getPlatform() !== 'android') return
  nuxtApp.hook('app:mounted', async () => {
    try {
      await SystemBars.hide()
    } catch (error) {
      console.error('Android fullscreen configuration failed', error)
    } finally {
      await SplashScreen.hide()
    }
  })
})
