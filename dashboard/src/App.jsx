import { useState } from 'react'
import { Terminal, Cpu, HardDrive, TestTube2, Settings, Zap } from 'lucide-react'

function App() {
  const [activeTab, setActiveTab] = useState('matrix')

  const navItems = [
    { id: 'matrix', label: 'МАТРИЦА ЗАДАЧ', icon: Terminal },
    { id: 'explorer', label: 'ХРАНИЛИЩЕ АРТЕФАКТОВ', icon: HardDrive },
    { id: 'lab', label: 'ЛАБОРАТОРИЯ ПРОМПТОВ', icon: TestTube2 },
    { id: 'hub', label: 'ВЕКТОРНЫЙ ХАБ', icon: Cpu },
  ]

  const activeLabel = navItems.find(item => item.id === activeTab)?.label || 'ПАНЕЛЬ'

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar - Industrial Terminal Look */}
      <aside className="w-64 border-r border-borderColor bg-surface flex flex-col">
        <div className="p-4 border-b border-borderColor flex items-center gap-2">
          <Zap className="text-primary w-5 h-5" />
          <h1 className="font-bold tracking-wider text-sm">НЕЙРОЦЕХ</h1>
        </div>
        
        <nav className="flex-1 p-4 space-y-2 font-mono text-sm">
          <div className="text-textMuted text-xs mb-4 uppercase tracking-widest">Модули</div>
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 text-left transition-colors ${
                activeTab === item.id 
                  ? 'bg-primary/10 text-primary border-l-2 border-primary' 
                  : 'text-textMuted hover:bg-surfaceHover hover:text-textMain border-l-2 border-transparent'
              }`}
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="p-4 border-t border-borderColor font-mono text-xs text-textMuted flex items-center gap-2">
          <Settings className="w-4 h-4" />
          <span>СИСТЕМА ГОТОВА</span>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col bg-background h-full overflow-hidden">
        {/* Header */}
        <header className="h-14 border-b border-borderColor flex items-center px-6 justify-between bg-surface/50 backdrop-blur">
          <h2 className="font-mono text-sm text-textMuted">
            {activeLabel}
          </h2>
          <div className="flex items-center gap-4 text-xs font-mono">
            <span className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
              9 АГЕНТОВ АКТИВНО
            </span>
            <span className="text-textMuted">Версия 4.6</span>
          </div>
        </header>

        {/* Dynamic Content */}
        <div className="flex-1 overflow-auto p-6">
          {activeTab === 'matrix' && <TaskMatrix />}
          {activeTab === 'explorer' && <OutputExplorer />}
          {(activeTab === 'lab' || activeTab === 'hub') && (
            <div className="flex items-center justify-center h-full border border-dashed border-borderColor text-textMuted font-mono text-sm">
              [ МОДУЛЬ В РАЗРАБОТКЕ ]
            </div>
          )}
        </div>
      </main>
    </div>
  )
}

function TaskMatrix() {
  const CHAR_LIMITS = {
    'free_style':  { min: 3000,  max: 30000, default: 10000, label: 'Свободная статья (Ручная настройка)' },
    'law_review':  { min: 4000,  max: 12000, default: 5000,  label: 'Разбор закона' },
    'case_study':  { min: 5500,  max: 20000, default: 8000,  label: 'Бизнес-кейс / Ситуация' },
    'checklist':   { min: 8000,  max: 25000, default: 12000, label: 'Чек-лист «10 пунктов»' },
    'analysis':    { min: 10000, max: 30000, default: 15000, label: 'Аналитический лонгрид' },
    'reference':   { min: 2000,  max: 10000, default: 4000,  label: 'Практический справочник' },
  }

  const MODELS_BY_PROVIDER = {
    deepseek: [
      { value: 'deepseek-v4-pro', label: 'DeepSeek v4 Pro (Рекомендуется)' },
      { value: 'deepseek-v4-flash', label: 'DeepSeek v4 Flash' }
    ],
    kie: [
      { value: 'claude-4.7', label: 'Claude 4.7' },
      { value: 'claude-4.6', label: 'Claude 4.6' },
      { value: 'gpt-5.5', label: 'GPT 5.5' }
    ],
    openai: [
      { value: 'gpt-4o', label: 'GPT-4o (Embeddings & Text)' },
      { value: 'o1', label: 'OpenAI o1' },
      { value: 'o3-mini', label: 'OpenAI o3-mini' }
    ]
  }

  const [topic, setTopic] = useState('')
  const [articleType, setArticleType] = useState('free_style')
  const [targetLength, setTargetLength] = useState(CHAR_LIMITS['free_style'].default)
  
  // Custom Freestyle / Article params
  const [description, setDescription] = useState('')
  const [styleNuances, setStyleNuances] = useState('')
  const [additionalInstructions, setAdditionalInstructions] = useState('')
  const [provider, setProvider] = useState('deepseek')
  const [model, setModel] = useState('deepseek-v4-pro')

  const handleTypeChange = (e) => {
    const newType = e.target.value
    setArticleType(newType)
    const limits = CHAR_LIMITS[newType]
    setTargetLength(limits.default)
  }

  const handleProviderChange = (e) => {
    const newProvider = e.target.value
    setProvider(newProvider)
    setModel(MODELS_BY_PROVIDER[newProvider][0].value)
  }

  const handleLaunchPipeline = () => {
    // Демонстрационный алерт с ТЗ
    alert(
      `Пайплайн успешно запущен с ТЗ:\n\n` +
      `Тема: ${topic || 'Не указана'}\n` +
      `Тип: ${articleType}\n` +
      `Объем: ~${Number(targetLength).toLocaleString()} символов\n` +
      `Провайдер: ${provider}\n` +
      `Модель: ${model}\n` +
      `Описание: ${description ? 'Присутствует' : 'Отсутствует'}\n` +
      `Стиль: ${styleNuances ? 'Присутствует' : 'По умолчанию'}\n` +
      `Доп. инструкции: ${additionalInstructions ? 'Присутствует' : 'Отсутствует'}`
    )
  }

  const limits = CHAR_LIMITS[articleType]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-6">
        {/* New Task Form */}
        <div className="col-span-2 border border-borderColor bg-surface p-5 transition-all">
          <h3 className="font-mono text-sm mb-4 border-b border-borderColor pb-2 flex justify-between items-center">
            <span>СОЗДАТЬ СТАТЬЮ (РУЧНАЯ НАСТРОЙКА)</span>
            <span className="text-primary font-mono text-xs">FREESTYLE MODE ACTIVE</span>
          </h3>
          
          <div className="grid grid-cols-2 gap-6">
            {/* Left column — core params */}
            <div className="space-y-4">
              <div>
                <label className="block font-mono text-xs text-textMuted mb-1">ТЕМА СТАТЬИ (Обязательно)</label>
                <input 
                  type="text" 
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                  className="w-full bg-background border border-borderColor p-2 text-sm focus:outline-none focus:border-primary text-textMain"
                  placeholder="Напр. Налоговая реформа 2026: выживание на УСН"
                />
              </div>
            
              <div>
                <label className="block font-mono text-xs text-textMuted mb-1">ФОРМАТ / ШАБЛОН</label>
                <select 
                  className="w-full bg-background border border-borderColor p-2 text-sm focus:outline-none focus:border-primary text-textMain"
                  value={articleType}
                  onChange={handleTypeChange}
                >
                  {Object.entries(CHAR_LIMITS).map(([key, val]) => (
                    <option key={key} value={key}>{val.label}</option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block font-mono text-xs text-textMuted mb-1">LLM ПРОВАЙДЕР</label>
                  <select 
                    className="w-full bg-background border border-borderColor p-2 text-sm focus:outline-none focus:border-primary text-textMain"
                    value={provider}
                    onChange={handleProviderChange}
                  >
                    <option value="deepseek">DeepSeek API</option>
                    <option value="kie">Kie.ai (Claude/GPT)</option>
                    <option value="openai">OpenAI (Embeddings)</option>
                  </select>
                </div>

                <div>
                  <label className="block font-mono text-xs text-textMuted mb-1">МОДЕЛЬ ГЕНЕРАЦИИ</label>
                  <select 
                    className="w-full bg-background border border-borderColor p-2 text-sm focus:outline-none focus:border-primary text-textMain"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                  >
                    {MODELS_BY_PROVIDER[provider].map((m) => (
                      <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <div className="flex justify-between font-mono text-xs text-textMuted mb-1">
                  <label>ЦЕЛЕВОЙ ОБЪЕМ (СИМВОЛЫ)</label>
                  <span className="text-primary font-bold">~{Number(targetLength).toLocaleString()}</span>
                </div>
                <input 
                  type="range" 
                  className="w-full accent-primary" 
                  min={limits.min}
                  max={limits.max}
                  step="1000"
                  value={targetLength}
                  onChange={(e) => setTargetLength(e.target.value)}
                />
                <div className="flex justify-between font-mono text-[10px] text-textMuted mt-1">
                  <span>{limits.min.toLocaleString()}</span>
                  <span>{limits.max.toLocaleString()}</span>
                </div>
              </div>

              <button 
                onClick={handleLaunchPipeline}
                className="w-full mt-4 bg-textMain text-background font-bold py-2.5 text-sm hover:bg-primary transition-colors uppercase tracking-wider"
              >
                Запустить Пайплайн
              </button>
            </div>

            {/* Right column — advanced custom inputs */}
            <div className="space-y-4 border-l border-borderColor pl-6">
              <div>
                <label className="block font-mono text-xs text-textMuted mb-1">
                  ОПИСАНИЕ СТАТЬИ / КОНТЕКСТ
                </label>
                <textarea 
                  rows="3"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full bg-background border border-borderColor p-2 text-xs focus:outline-none focus:border-primary text-textMain font-sans resize-none"
                  placeholder="Вставьте контекст статьи, целевую аудиторию, ключевые факты, структуру или сырые данные..."
                />
              </div>

              <div>
                <label className="block font-mono text-xs text-textMuted mb-1">
                  ПОЖЕЛАНИЯ К СТИЛЮ И ТОНАЛЬНОСТИ
                </label>
                <textarea 
                  rows="2"
                  value={styleNuances}
                  onChange={(e) => setStyleNuances(e.target.value)}
                  className="w-full bg-background border border-borderColor p-2 text-xs focus:outline-none focus:border-primary text-textMain font-sans resize-none"
                  placeholder="Напр. Использовать жесткие бизнес-метафоры, избегать глянца, больше приземления, плотный ритм..."
                />
              </div>

              <div>
                <label className="block font-mono text-xs text-textMuted mb-1">
                  ДОПОЛНИТЕЛЬНЫЕ ИНСТРУКЦИИ
                </label>
                <textarea 
                  rows="2"
                  value={additionalInstructions}
                  onChange={(e) => setAdditionalInstructions(e.target.value)}
                  className="w-full bg-background border border-borderColor p-2 text-xs focus:outline-none focus:border-primary text-textMain font-sans resize-none"
                  placeholder="Особые требования к оформлению блоков, формул, законов или ссылок на реестры..."
                />
              </div>
            </div>
          </div>
        </div>

        {/* Active Processes Queue */}
        <div className="col-span-1 border border-borderColor bg-surface p-5 flex flex-col transition-all">
          <h3 className="font-mono text-sm mb-4 border-b border-borderColor pb-2">АКТИВНЫЕ ПРОЦЕССЫ</h3>
          
          <div className="flex-1 bg-background border border-borderColor p-4 font-mono text-xs overflow-auto space-y-2">
            <div className="text-textMuted">2026-05-14 14:27:29 | ИНФО | <span className="text-primary">СТАРТ</span> Задача: НДС для общепита 2026</div>
            <div className="text-textMuted">2026-05-14 14:27:31 | ИНФО | 🔎 Разведчик: Анализ трендов в сети...</div>
            <div className="text-warning">2026-05-14 14:27:35 | ВНИМАНИЕ | ⚖️ Исследователь: Коллизия в ставках НДС (10% vs 20%)</div>
            <div className="text-textMain">2026-05-14 14:27:40 | ИНФО | 🏗️ Инженер: Чертеж готов (7 блоков H2)</div>
            <div className="flex items-center gap-2 mt-2">
              <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
              <span className="text-textMain">❤️ Писатель: Написание секции 3/7...</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}



function OutputExplorer() {
  const [selectedFile, setSelectedFile] = useState('article.md')
  
  return (
    <div className="h-full flex flex-col border border-borderColor bg-surface">
      <div className="border-b border-borderColor p-4 flex justify-between items-center bg-surfaceHover">
        <h3 className="font-mono text-sm">Папка: /output/20260514_1427_nds_obshchepit</h3>
        <div className="flex gap-2">
          <button className="px-3 py-1 bg-primary text-background font-mono text-xs hover:bg-opacity-80">Опубликовать в блог</button>
          <button className="px-3 py-1 border border-borderColor font-mono text-xs hover:bg-background">Скачать ZIP</button>
        </div>
      </div>
      
      <div className="flex-1 flex overflow-hidden">
        {/* File Tree */}
        <div className="w-48 border-r border-borderColor p-4 bg-background font-mono text-xs">
          <div className="text-textMuted mb-2 uppercase">Артефакты</div>
          <ul className="space-y-2">
            {['article.md', 'seo_package.json', 'schema.json', 'pipeline_debug.json'].map(file => (
              <li 
                key={file}
                onClick={() => setSelectedFile(file)}
                className={`cursor-pointer hover:text-primary ${selectedFile === file ? 'text-primary' : 'text-textMain'}`}
              >
                ├── {file}
              </li>
            ))}
          </ul>
        </div>
        
        {/* File Content Preview */}
        <div className="flex-1 p-6 overflow-auto bg-background/50">
          {selectedFile === 'article.md' && (
            <div className="prose prose-invert max-w-none">
              <h1 className="text-3xl font-bold font-sans mb-6">Налоговая мясорубка: Как общепиту выжить с новым НДС в 2026 году</h1>
              <p className="text-textMuted font-mono text-xs border-l-2 border-primary pl-4 mb-8">
                [Время: Конец квартала. Атмосфера: Тревожное ожидание. Пространство: Бухгалтерия ресторана.]
              </p>
              <p>Штраф в 40% от неуплаченной суммы — это не страшилка из 90-х, а реальная перспектива для владельцев кафе в 2026 году, которые проигнорируют новые правила игры.</p>
              <h2 className="text-xl font-bold mt-8 mb-4">Анатомия изменений: Откуда взялся НДС на УСН</h2>
              <div className="bg-surface border border-borderColor p-4 my-4 font-mono text-sm text-textMuted">
                НАЖИВКА ДЛЯ ЦИТИРОВАНИЯ: <br/>
                <span className="text-textMain">С 1 января 2026 года предприятия общепита на УСН с доходом свыше 60 млн рублей обязаны платить НДС. Базовая ставка составляет 20%, однако предусмотрены льготные варианты: 5% без права на вычеты при доходе до 250 млн рублей и 7% при доходе до 400 млн рублей.</span>
              </div>
            </div>
          )}
          {selectedFile === 'pipeline_debug.json' && (
            <pre className="font-mono text-xs text-warning">
{`{
  "agent_metrics": {
    "heart_iterations": 2,
    "sheriff_rejections": 1,
    "mirror_ai_score": 12.4,
    "final_turing_score": 94.5
  },
  "sheriff_log": [
    "ОТКЛОНЕНО: Обнаружен H2 блок 'Вместо вывода'. Нарушено правило 'Смерть Заключениям'.",
    "ПРИНЯТО: Заменено на призыв к действию (CTA)."
  ]
}`}
            </pre>
          )}
          {selectedFile !== 'article.md' && selectedFile !== 'pipeline_debug.json' && (
             <div className="flex items-center justify-center h-full text-textMuted font-mono text-xs">
                [ ВЫБЕРИТЕ ФАЙЛ ДЛЯ ПРОСМОТРА ]
             </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
