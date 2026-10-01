import { mount } from '@vue/test-utils'
import HadithResultCard from '~/components/HadithResultCard.vue'
import { hadithFixture } from '../fixtures/hadith'

describe('HadithResultCard', () => {
  it('shows the source text and emits the selected hadith', async () => {
    const wrapper = mount(HadithResultCard, { props: { hadith: hadithFixture, position: 1 } })
    expect(wrapper.text()).toContain(hadithFixture.title)
    expect(wrapper.get('.excerpt').text()).toBe(hadithFixture.translation)
    expect(wrapper.get('.result-number').text()).toBe('01')
    await wrapper.get('button').trigger('click')
    expect(wrapper.emitted('read')?.[0]).toEqual([hadithFixture])
    expect(wrapper.get('a').attributes('href')).toBe(hadithFixture.source_url)
    expect(wrapper.get('a').attributes('rel')).toBe('noopener noreferrer')
  })

  it('omits a missing grade and displays markup as plain text', () => {
    const wrapper = mount(HadithResultCard, {
      props: {
        hadith: { ...hadithFixture, grade: null, title: '<img src=x onerror=alert(1)>' },
        position: 2,
      },
    })
    expect(wrapper.find('.grade').exists()).toBe(false)
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.get('h3').text()).toContain('<img')
  })
})
