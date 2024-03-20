import { reactive } from 'vue'

export default () => {
  const clippingNotice = reactive({
    type: null as string | null,
    visible: false,
  })

  return {
    clippingNotice,
    toggleClippingNotice(type: string | null) {
      if (type === null) {
        // closes popover without removing content
        clippingNotice.visible = false
      } else {
        clippingNotice.type = type
        clippingNotice.visible = true
      }
    },
  }
}
