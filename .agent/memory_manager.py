import asyncio
import os
from typing import List, Optional
from notebooklm import NotebookLMClient

class ProjectMemory:
    """
    Класс для управления контекстом и памятью проекта через Google NotebookLM.
    Автоматически создает Notebook для проекта и загружает все нужные файлы контекста.
    """
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.notebook_id: Optional[str] = None
        
    async def initialize(self, context_files: List[str] = None) -> str:
        """Инициализирует память (создает блокнот или находит существующий) и загружает файлы."""
        if context_files is None:
            context_files = []
            
        async with await NotebookLMClient.from_storage() as client:
            print(f"[Memory] Инициализация памяти для проекта '{self.project_name}'...")
            
            # Проверяем, есть ли уже блокнот с таким именем
            notebooks = await client.notebooks.list()
            target_notebook = next((nb for nb in notebooks if nb.title == self.project_name), None)
            
            if not target_notebook:
                target_notebook = await client.notebooks.create(self.project_name)
                print(f"[Memory] Создан новый Notebook: {target_notebook.id}")
            else:
                print(f"[Memory] Найдена существующая память проекта: {target_notebook.id}")
            
            self.notebook_id = target_notebook.id
            
            # Получаем список уже загруженных источников
            existing_sources = await client.sources.list(self.notebook_id)
            existing_titles = [src.title for src in existing_sources]
            
            # Загружаем файлы, которых еще нет в Notebook
            for file_path in context_files:
                if not os.path.exists(file_path):
                    print(f"[Memory] Файл не найден: {file_path}")
                    continue
                    
                filename = os.path.basename(file_path)
                if filename not in existing_titles:
                    print(f"[Memory] Загрузка локального контекста: {file_path}")
                    try:
                        await client.sources.add_file(self.notebook_id, file_path, wait=True)
                    except Exception as e:
                        print(f"[Memory] Ошибка при загрузке {file_path}: {e}")
                else:
                    print(f"[Memory] Файл уже в контексте: {filename}")
                    
            print("[Memory] Инициализация завершена.")
            return self.notebook_id
            
    async def ask(self, query: str) -> str:
        """Задает вопрос по контексту проекта."""
        if not self.notebook_id:
            raise ValueError("Память не инициализирована. Сначала вызовите initialize().")
            
        async with await NotebookLMClient.from_storage() as client:
            print(f"[Memory] Запрос к памяти: '{query}'...")
            response = await client.chat.ask(self.notebook_id, query)
            return response.answer
            
    async def save_context(self, important_info: str):
        """Сохраняет важную информацию как отдельную заметку внутри проекта."""
        if not self.notebook_id:
            raise ValueError("Память не инициализирована.")
            
        async with await NotebookLMClient.from_storage() as client:
            # Для сохранения используем chat ask и просим превратить это в note
            prompt = f"Пожалуйста, сохрани эту важную информацию в контекст проекта:\n\n{important_info}"
            await client.chat.ask(self.notebook_id, prompt, save_as_note=True)
            print("[Memory] Информация сохранена в контекст.")

# Пример использования (можно раскомментировать для теста)
async def test_memory():
    # 1. Задаем имя проекта (под этим именем будет создан NotebookLM)
    memory = ProjectMemory("Smart Agent Context")
    
    # 2. Передаем список файлов (инструкции, правила, архитектура)
    context_files = ["README.md"]
    
    await memory.initialize(context_files)
        
    # 3. Делаем запрос к памяти
    answer = await memory.ask("О чем этот проект? Сделай краткое саммари.")
    print(f"\nОтвет:\n{answer}")

if __name__ == "__main__":
    # asyncio.run(test_memory())
    pass
