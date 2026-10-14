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
}

export default config
