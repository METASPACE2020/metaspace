export default (values: string[]): string => {
  const escaped = values.map(v => {
    if (v != null) {
      return `"${String(v).replace(/"/g, '""')}"`
    } else {
      return ''
    }
  })

  return escaped.join(',') + '\n'
}

export const csvExportHeader = () => {
  const dateStr = new Date().toLocaleString().replace(/,/g, '')
  return `# Generated at ${dateStr}. For help see https://bit.ly/3Bzs6Z4\n`
  + `# URL: ${window.location.href}\n`
}

/**
 * For arrays of text values, primarily molecule names, follow the pattern defined in /docs/csv_export.md for
 * unambiguously encoding list items: Separate items with `, `, and ensure that items do not contain `, `
 * inside their value. That way they can be separated again with `.split(', ')`.
 * This is a lossy transformation intended to handle the most common use cases, so don't use it if data integrity is a
 * strict requirement
 */
export const formatCsvTextArray = (values: string[]): string =>
  values
    .map(val => (val ?? '').replace(/, +/g, ','))
    .join(', ')
