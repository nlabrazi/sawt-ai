import { mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'

import { useMiniToast } from '~/composables/useMiniToast'

const ToastHarness = defineComponent({
  setup() {
    return useMiniToast(1000)
  },
  template: `
    <button type="button" @click="show('Action confirmée')">Afficher</button>
    <span v-if="visible" class="message">{{ message }}</span>
  `,
})

describe('useMiniToast', () => {
  it('automatically dismisses the current message', async () => {
    vi.useFakeTimers()
    const wrapper = mount(ToastHarness)

    await wrapper.get('button').trigger('click')

    expect(wrapper.get('.message').text()).toBe('Action confirmée')

    vi.advanceTimersByTime(1000)
    await nextTick()

    expect(wrapper.find('.message').exists()).toBe(false)
  })
})
