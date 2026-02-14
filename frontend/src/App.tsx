import { Button } from '@/components/ui/Button'

function App() {
  return (
    <div style={{ padding: 40 }}>
      <Button>Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="outline">Outline</Button>
      <Button loading>Loading</Button>
    </div>
  )
}

export default App
