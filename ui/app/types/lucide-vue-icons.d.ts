declare module '@lucide/vue/dist/esm/icons/*.mjs' {
  import type { FunctionalComponent, SVGAttributes } from 'vue'

  type LucideIconProps = SVGAttributes & {
    absoluteStrokeWidth?: boolean
    color?: string
    size?: number | string
    strokeWidth?: number | string
  }

  const icon: FunctionalComponent<LucideIconProps>

  export default icon
}
