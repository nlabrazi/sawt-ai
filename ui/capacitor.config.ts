import type { CapacitorConfig } from '@capacitor/cli'

const config: CapacitorConfig = {
  appId: 'dev.nabster.sawtai',
  appName: 'Sawt AI',
  webDir: '.output-android/public',
  server: {
    hostname: 'localhost',
    androidScheme: 'https',
    cleartext: false,
  },
  android: {
    allowMixedContent: false,
    webContentsDebuggingEnabled: false,
    backgroundColor: '#07101d',
  },
  plugins: {
    SystemBars: {
      insetsHandling: 'native',
      initialViewportFitValueHint: 'cover',
      hidden: true,
      style: 'DARK',
    },
    SplashScreen: {
      launchAutoHide: false,
      backgroundColor: '#07101d',
      androidSplashResourceName: 'sawt_splash',
      showSpinner: false,
    },
  },
}

export default config
