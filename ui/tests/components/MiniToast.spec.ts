import { mount } from '@vue/test-utils'

import MiniToast from '~/components/MiniToast.vue'

describe('MiniToast', () => {
  it('announces a compact success message when open', async () => {
    const wrapper = mount(MiniToast, {
      props: {
        open: true,
        message: 'Verset copié',
      },
      global: {
        stubs: {
          teleport: true,
        },
      },
    })

    const toast = wrapper.get('.mini-toast')

    expect(toast.text()).toBe('Verset copié')
    expect(toast.attributes('role')).toBe('status')
    expect(toast.attributes('aria-live')).toBe('polite')
    expect(wrapper.find('.lucide-circle-check').exists()).toBe(true)

    await wrapper.setProps({ open: false })

    expect(wrapper.find('.mini-toast').exists()).toBe(false)
  })
})
