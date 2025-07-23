import os
import ast
import json
import re
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import aiofiles
from datetime import datetime
import logging
from pathlib import Path

# Vector search and embeddings
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logger.warning("Vector search dependencies not available. Install sentence-transformers and chromadb.")

from src.core.models import CodeSearchResult, FileState

logger = logging.getLogger(__name__)

class CodebaseAwarenessSystem:
    """
    Advanced codebase search and awareness system with semantic search,
    AST analysis, and contextual code understanding.
    """
    
    def __init__(self, workspace_path: str = "./workspace"):
        self.workspace_path = workspace_path
        self.file_index: Dict[str, Dict] = {}
        self.function_index: Dict[str, List[Dict]] = {}
        self.class_index: Dict[str, List[Dict]] = {}
        self.import_graph: Dict[str, List[str]] = {}
        
        # Vector search setup
        self.vector_db = None
        self.embedder = None
        if VECTOR_SEARCH_AVAILABLE:
            self._setup_vector_search()
        
        # Supported file extensions for analysis
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby'
        }
    
    def _setup_vector_search(self):
        """Initialize vector search components"""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
            self.vector_db = self.chroma_client.get_or_create_collection(
                name="codebase",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Initialize sentence transformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Vector search initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector search: {e}")
            self.vector_db = None
            self.embedder = None
    
    async def index_workspace(self):
        """Index the entire workspace for search and awareness"""
        logger.info(f"Indexing workspace: {self.workspace_path}")
        
        for root, dirs, files in os.walk(self.workspace_path):
            # Skip common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
            
            for file in files:
                if file.startswith('.'):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.workspace_path)
                
                try:
                    await self._index_file(relative_path)
                except Exception as e:
                    logger.error(f"Error indexing file {relative_path}: {e}")
        
        logger.info(f"Indexing complete. Indexed {len(self.file_index)} files")
    
    async def _index_file(self, file_path: str):
        """Index a single file"""
        full_path = os.path.join(self.workspace_path, file_path)
        
        if not os.path.exists(full_path):
            return
        
        # Get file info
        stat = os.stat(full_path)
        file_info = {
            'path': file_path,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': Path(file_path).suffix.lower(),
            'language': self._detect_language(file_path)
        }
        
        # Read file content
        try:
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            file_info['lines'] = len(content.splitlines())
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return
        
        # Analyze based on language
        if file_info['language'] == 'python':
            await self._analyze_python_file(file_path, content, file_info)
        elif file_info['language'] in ['javascript', 'typescript']:
            await self._analyze_js_file(file_path, content, file_info)
        else:
            await self._analyze_generic_file(file_path, content, file_info)
        
        # Store in index
        self.file_index[file_path] = file_info
        
        # Add to vector database
        if self.vector_db and self.embedder:
            await self._add_to_vector_db(file_path, content, file_info)
    
    async def _analyze_python_file(self, file_path: str, content: str, file_info: Dict):
        """Analyze Python file using AST"""
        try:
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'docstring': ast.get_docstring(node),
                        'file': file_path
                    }
                    functions.append(func_info)
                    
                    # Add to global function index
                    if node.name not in self.function_index:
                        self.function_index[node.name] = []
                    self.function_index[node.name].append(func_info)
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node),
                        'file': file_path
                    }
                    classes.append(class_info)
                    
                    # Add to global class index
                    if node.name not in self.class_index:
                        self.class_index[node.name] = []
                    self.class_index[node.name].append(class_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    else:
                        if node.module:
                            imports.append(node.module)
            
            file_info.update({
                'functions': functions,
                'classes': classes,
                'imports': imports
            })
            
            # Update import graph
            self.import_graph[file_path] = imports
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing Python file {file_path}: {e}")
    
    async def _analyze_js_file(self, file_path: str, content: str, file_info: Dict):
        """Basic JavaScript/TypeScript analysis using regex patterns"""
        # Find functions
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'const\s+(\w+)\s*=\s*\(',
            r'let\s+(\w+)\s*=\s*\(',
            r'var\s+(\w+)\s*=\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=>\s*'
        ]
        
        functions = []
        for pattern in function_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                func_name = match.group(1)
                line_num = content[:match.start()].count('\n') + 1
                
                func_info = {
                    'name': func_name,
                    'line': line_num,
                    'file': file_path
                }
                functions.append(func_info)
                
                # Add to global function index
                if func_name not in self.function_index:
                    self.function_index[func_name] = []
                self.function_index[func_name].append(func_info)
        
        # Find imports
        import_patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]',
            r'import\s+[\'"]([^\'"]+)[\'"]',
            r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
        ]
        
        imports = []
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                imports.append(match.group(1))
        
        file_info.update({
            'functions': functions,
            'imports': imports
        })
        
        self.import_graph[file_path] = imports
    
    async def _analyze_generic_file(self, file_path: str, content: str, file_info: Dict):
        """Basic analysis for unsupported file types"""
        # Just count some basic metrics
        file_info.update({
            'word_count': len(content.split()),
            'char_count': len(content),
            'functions': [],
            'classes': [],
            'imports': []
        })
    
    async def _add_to_vector_db(self, file_path: str, content: str, file_info: Dict):
        """Add file content to vector database for semantic search"""
        try:
            # Split content into chunks for better search
            chunks = self._split_content(content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_path}:chunk_{i}"
                
                # Create embedding
                embedding = self.embedder.encode([chunk])[0].tolist()
                
                # Add to vector database
                self.vector_db.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        'file_path': file_path,
                        'chunk_index': i,
                        'language': file_info.get('language', 'unknown'),
                        'line_start': i * 50 + 1,  # Approximate line numbers
                        'line_end': min((i + 1) * 50, file_info.get('lines', 0))
                    }]
                )
                
        except Exception as e:
            logger.error(f"Error adding {file_path} to vector database: {e}")
    
    def _split_content(self, content: str, chunk_size: int = 500) -> List[str]:
        """Split content into overlapping chunks for better search"""
        lines = content.splitlines()
        chunks = []
        
        current_chunk = []
        current_size = 0
        
        for line in lines:
            current_chunk.append(line)
            current_size += len(line)
            
            if current_size >= chunk_size:
                chunks.append('\n'.join(current_chunk))
                # Keep some overlap
                current_chunk = current_chunk[-10:] if len(current_chunk) > 10 else []
                current_size = sum(len(line) for line in current_chunk)
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        return self.supported_extensions.get(ext, 'text')
    
    async def search_code(self, query: str, file_type: Optional[str] = None,
                         max_results: int = 10) -> List[CodeSearchResult]:
        """Search codebase using multiple strategies"""
        results = []
        
        # 1. Semantic search (if available)
        if self.vector_db and self.embedder:
            vector_results = await self._semantic_search(query, max_results // 2)
            results.extend(vector_results)
        
        # 2. Text-based search
        text_results = await self._text_search(query, file_type, max_results // 2)
        results.extend(text_results)
        
        # 3. Function/class name search
        name_results = await self._name_search(query, max_results // 4)
        results.extend(name_results)
        
        # Deduplicate and sort by relevance
        seen = set()
        unique_results = []
        for result in sorted(results, key=lambda r: r.score, reverse=True):
            key = (result.file_path, result.line_number)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results[:max_results]
    
    async def _semantic_search(self, query: str, max_results: int) -> List[CodeSearchResult]:
        """Semantic search using vector embeddings"""
        if not (self.vector_db and self.embedder):
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedder.encode([query])[0].tolist()
            
            # Search vector database
            results = self.vector_db.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )
            
            search_results = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                document = results['documents'][0][i]
                distance = results['distances'][0][i] if 'distances' in results else 0.0
                
                # Convert distance to similarity score
                score = 1.0 - distance
                
                result = CodeSearchResult(
                    file_path=metadata['file_path'],
                    line_number=metadata['line_start'],
                    content=document,
                    score=score
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    async def _text_search(self, query: str, file_type: Optional[str], 
                          max_results: int) -> List[CodeSearchResult]:
        """Traditional text-based search"""
        results = []
        query_lower = query.lower()
        
        for file_path, file_info in self.file_index.items():
            if file_type and file_info.get('language') != file_type:
                continue
            
            try:
                full_path = os.path.join(self.workspace_path, file_path)
                async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                
                lines = content.splitlines()
                for line_num, line in enumerate(lines, 1):
                    if query_lower in line.lower():
                        # Calculate simple relevance score
                        score = query_lower.count(' ') + 1 if ' ' in query_lower else 1
                        score *= (query_lower.count(query_lower) / len(line)) if line else 0
                        
                        # Get context
                        context_before = lines[max(0, line_num-3):line_num-1]
                        context_after = lines[line_num:min(len(lines), line_num+3)]
                        
                        result = CodeSearchResult(
                            file_path=file_path,
                            line_number=line_num,
                            content=line,
                            context_before=context_before,
                            context_after=context_after,
                            score=score
                        )
                        results.append(result)
                        
                        if len(results) >= max_results:
                            break
                            
            except Exception as e:
                logger.error(f"Error searching in file {file_path}: {e}")
        
        return results
    
    async def _name_search(self, query: str, max_results: int) -> List[CodeSearchResult]:
        """Search for function and class names"""
        results = []
        query_lower = query.lower()
        
        # Search functions
        for func_name, func_infos in self.function_index.items():
            if query_lower in func_name.lower():
                for func_info in func_infos:
                    score = 1.0 if func_name.lower() == query_lower else 0.7
                    result = CodeSearchResult(
                        file_path=func_info['file'],
                        line_number=func_info['line'],
                        content=f"def {func_name}({', '.join(func_info.get('args', []))})",
                        score=score
                    )
                    results.append(result)
        
        # Search classes
        for class_name, class_infos in self.class_index.items():
            if query_lower in class_name.lower():
                for class_info in class_infos:
                    score = 1.0 if class_name.lower() == query_lower else 0.7
                    result = CodeSearchResult(
                        file_path=class_info['file'],
                        line_number=class_info['line'],
                        content=f"class {class_name}:",
                        score=score
                    )
                    results.append(result)
        
        return sorted(results, key=lambda r: r.score, reverse=True)[:max_results]
    
    async def get_file_summary(self, file_path: str) -> Optional[Dict]:
        """Get comprehensive summary of a file"""
        if file_path not in self.file_index:
            await self._index_file(file_path)
        
        return self.file_index.get(file_path)
    
    async def get_function_info(self, function_name: str) -> List[Dict]:
        """Get information about all functions with the given name"""
        return self.function_index.get(function_name, [])
    
    async def get_class_info(self, class_name: str) -> List[Dict]:
        """Get information about all classes with the given name"""
        return self.class_index.get(class_name, [])
    
    async def get_import_dependencies(self, file_path: str) -> List[str]:
        """Get files that this file imports"""
        return self.import_graph.get(file_path, [])
    
    async def get_dependent_files(self, file_path: str) -> List[str]:
        """Get files that import this file"""
        dependents = []
        file_name = Path(file_path).stem
        
        for importing_file, imports in self.import_graph.items():
            for imp in imports:
                if file_name in imp or file_path in imp:
                    dependents.append(importing_file)
        
        return dependents
    
    async def get_project_overview(self) -> Dict[str, Any]:
        """Get high-level overview of the project"""
        total_files = len(self.file_index)
        total_functions = sum(len(funcs) for funcs in self.function_index.values())
        total_classes = sum(len(classes) for classes in self.class_index.values())
        
        # Language breakdown
        languages = {}
        for file_info in self.file_index.values():
            lang = file_info.get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        # Most connected files (high import/dependency count)
        connected_files = []
        for file_path in self.file_index.keys():
            import_count = len(self.import_graph.get(file_path, []))
            dependent_count = len(await self.get_dependent_files(file_path))
            total_connections = import_count + dependent_count
            
            if total_connections > 0:
                connected_files.append((file_path, total_connections))
        
        connected_files.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_files': total_files,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'languages': languages,
            'most_connected_files': connected_files[:10],
            'indexed_at': datetime.now().isoformat()
        }
