import re
import json
import copy
import os
from typing import List, Dict, Any, Optional, Tuple, Callable, Union, Set
import pymysql

class EnhancedMarkdownParser:
    """
    增强版Markdown解析器，结合dify-rag的markdown解析和splitter方法
    具有表格中文描述化和智能分块能力
    """
    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        contain_closest_title_levels: int = 2,
        convert_table_to_text: bool = True,
        separators: List[str] = None,
        remove_hyperlinks: bool = False,
        remove_images: bool = False,
        encoding: str = "utf-8",
        allow_cross_header_content: bool = False,
        db_config: Optional[Dict[str, str]] = None,
        target_db_extension: str = ".pdf"
    ):
        if chunk_overlap >= max_chunk_size:
            raise ValueError(f"重叠大小 ({chunk_overlap}) 必须小于块大小 ({max_chunk_size})")

        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.contain_closest_title_levels = contain_closest_title_levels
        self.convert_table_to_text = convert_table_to_text
        self.separators = separators if separators else ["\n\n", "\n", " ", ""]
        self.default_remove_hyperlinks = remove_hyperlinks
        self.default_remove_images = remove_images
        self.encoding = encoding
        self.allow_cross_header_content = allow_cross_header_content
        self.target_db_extension = target_db_extension

        self.doc_title = "" # 用于显示和块内标题的文档标题
        self.chunks = []
        self.chinese_numbers = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
        for i in range(11, 101): # 扩展中文数字
            if i <= 19: self.chinese_numbers.append(f"十{self.chinese_numbers[i-11] if i > 10 else ''}")
            elif i % 10 == 0: self.chinese_numbers.append(f"{self.chinese_numbers[i//10-1]}十")
            else: self.chinese_numbers.append(f"{self.chinese_numbers[i//10-1]}十{self.chinese_numbers[i%10-1]}")

        self.chunk_counter = 0
        self.original_filename_for_db_lookup = None # 用于数据库查询的（转换后的或指定的）文件名

        self.db_config = db_config
        self.db_conn = None
        self.db_cursor = None
        self.db_metadata_for_current_doc = {} # 存储当前文档从数据库获取的元数据
        self._connect_db()

    def _connect_db(self) -> None:
        if self.db_config:
            try:
                self.db_conn = pymysql.connect(**self.db_config, cursorclass=pymysql.cursors.DictCursor)
                self.db_cursor = self.db_conn.cursor()
                print("成功连接到数据库。")
            except pymysql.MySQLError as err:
                print(f"连接数据库时出错: {err}")
                self.db_conn = None
                self.db_cursor = None
        else:
            print("未提供数据库配置，跳过数据库连接。")

    def close_db_connection(self) -> None:
        if self.db_cursor:
            self.db_cursor.close()
        if self.db_conn and self.db_conn.open:
            self.db_conn.close()
            print("数据库连接已关闭。")
        elif self.db_conn is None and self.db_cursor is None and self.db_config is None:
            pass # No connection was attempted
        else:
            print("数据库连接已关闭或未成功打开。")


    def _fetch_db_metadata(self, filename_key: str) -> Dict[str, Any]:
        """根据提供的filename_key从数据库获取元数据"""
        if not self.db_cursor:
            return {}
        try:
            # 确保表名和列名与您的数据库模式匹配
            query = "SELECT resourceId, dcId, resourceName, originFileName FROM Rag_dataBase WHERE resourceName = %s"
            self.db_cursor.execute(query, (filename_key,))
            result = self.db_cursor.fetchone()
            if result:
                print(f"成功为 '{filename_key}' 获取到数据库元数据。")
                return result
            else:
                print(f"未找到 resourceName 为 '{filename_key}' 的数据库元数据。")
                return {}
        except pymysql.MySQLError as err:
            print(f"查询 '{filename_key}' 的数据库时出错: {err}")
            return {}
        except Exception as e:
            print(f"为 '{filename_key}' 获取数据库元数据时发生意外错误: {e}")
            return {}

    def parse_file(
        self,
        file_path: str, # 例如 "D:\folder\xxx.md"
        doc_title: Optional[str] = None,
        remove_hyperlinks: Optional[bool] = None,
        remove_images: Optional[bool] = None,
        db_resource_name_for_metadata: Optional[str] = None # 新增参数：直接指定用于数据库查询的resourceName
    ) -> List[Dict]:
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                markdown_content = f.read()

            base_filename_without_ext = os.path.splitext(os.path.basename(file_path))[0]

            # 决定用于数据库查询的文件名
            if db_resource_name_for_metadata:
                # 如果用户提供了明确的数据库resourceName，则使用它
                filename_for_db_lookup = db_resource_name_for_metadata
                print(f"使用用户提供的数据库 resourceName 进行元数据查询: '{filename_for_db_lookup}'")
            else:
                # 否则，使用从输入文件名和 target_db_extension 推断的名称
                filename_for_db_lookup = base_filename_without_ext + self.target_db_extension
                print(f"用于数据库查询的文件名已转换为: '{filename_for_db_lookup}' (基于输入 '{os.path.basename(file_path)}')")

            # doc_title_param 用于块元数据中的 'doc_title' 和生成 chunk_id
            # 它仍然基于传入的 doc_title 或 .md 文件名
            doc_title_param = doc_title if doc_title else base_filename_without_ext

            return self.parse(
                markdown_content,
                doc_title_param, # 这个是显示标题
                remove_hyperlinks,
                remove_images,
                original_filename_for_db_lookup=filename_for_db_lookup # 这个是用于数据库查找的键
            )
        except FileNotFoundError:
            print(f"错误: 文件未找到 {file_path}")
            return []
        except Exception as e:
            print(f"读取文件 '{file_path}' 时出错: {str(e)}")
            return []

    def parse(
        self,
        markdown: str,
        doc_title: str = "Untitled",
        remove_hyperlinks: Optional[bool] = None,
        remove_images: Optional[bool] = None,
        original_filename_for_db_lookup: Optional[str] = None
    ) -> List[Dict]:
        self.doc_title = doc_title # 这是显示标题，例如 "xxx"
        self.original_filename_for_db_lookup = original_filename_for_db_lookup if original_filename_for_db_lookup else doc_title
        self.chunks = []
        self.chunk_counter = 0

        if self.original_filename_for_db_lookup:
            self.db_metadata_for_current_doc = self._fetch_db_metadata(self.original_filename_for_db_lookup)
        else:
            self.db_metadata_for_current_doc = {}
            print("警告：未提供用于数据库查询的（转换后或指定的）文件名。")

        _remove_hyperlinks = self.default_remove_hyperlinks if remove_hyperlinks is None else remove_hyperlinks
        _remove_images = self.default_remove_images if remove_images is None else remove_images

        if _remove_hyperlinks: markdown = self._remove_hyperlinks(markdown)
        if _remove_images: markdown = self._remove_images(markdown)

        markdown_text_remainder, md_table_strs, html_table_strs = self._extract_tables_and_remainder(markdown)
        
        document_sections = self._markdown_to_sections(markdown_text_remainder)
        
        for hierarchy_headers, section_content in document_sections:
            if not section_content.strip():
                continue
            section_title = self._extract_section_title(hierarchy_headers)
            self._process_section_content(section_content, section_title)
            
        for table_md_str in md_table_strs:
            if table_md_str.strip():
                self._process_markdown_table(table_md_str, "文档级独立Markdown表格")
        
        for html_table_str_item in html_table_strs:
            if html_table_str_item.strip():
                self._process_html_table_str(html_table_str_item, "文档级独立HTML表格")
        
        return self.chunks

    def _add_chunk(self, content: str, metadata: Dict[str, Any]) -> None:
        self.chunk_counter += 1
        doc_name_slug = re.sub(r'\s+', '_', self.doc_title)
        doc_name_slug = re.sub(r'[^\w\-_]', '', doc_name_slug)
        chunk_id = f"{doc_name_slug}_{self.chunk_counter}"

        final_metadata = {
            "chunk_id": chunk_id,
            "doc_title": self.doc_title, # 显示/输入标题
            **metadata
        }
        if self.db_metadata_for_current_doc:
            final_metadata.update(self.db_metadata_for_current_doc)

        # 决定在块的内容标题中使用哪个标题。
        # 优先使用 'originFileName' (如果从DB获取到且非空)。
        # 否则回退到 'doc_title' (显示/输入标题)。
        title_for_content_header = final_metadata.get('originFileName')
        if not title_for_content_header: # 检查 originFileName 是否为 None 或空字符串
            title_for_content_header = final_metadata.get('doc_title', 'Untitled') # 回退

        section_title_for_header = final_metadata.get('section_title', 'Section')
        
        header_text = f"# {title_for_content_header} --- {section_title_for_header}\n\n"
        self.chunks.append({"content": header_text + content, "metadata": final_metadata})

    def _remove_hyperlinks(self, content: str) -> str:
        return re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", content)

    def _remove_images(self, content: str) -> str:
        content = re.sub(r"!{1}\[\[(.*?)\]\]", "", content)
        return re.sub(r"!\[(.*?)\]\((.*?)\)", "", content)

    def _extract_tables_and_remainder(self, markdown_text: str) -> Tuple[str, List[str], List[str]]:
        remainder = markdown_text
        html_table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)
        html_tables = html_table_pattern.findall(remainder)
        remainder = html_table_pattern.sub('<!-- HTML_TABLE_PLACEHOLDER -->', remainder)

        md_table_pattern = re.compile(
            r"(?:\n|^)"
            r"((?:\|.*?\|[ \t]*\n)+"
            r"(?:\|(?:\s*[:-]+[-| :]*\s*)\|.*?\n)"
            r"(?:\|.*?\|[ \t]*\n)+)",
            re.VERBOSE
        )
        md_tables_found = md_table_pattern.findall(remainder)
        md_tables = [match[0] if isinstance(match, tuple) else match for match in md_tables_found]
        remainder = md_table_pattern.sub("\n<!-- MD_TABLE_PLACEHOLDER -->\n", remainder)
        
        remainder = remainder.replace('<!-- HTML_TABLE_PLACEHOLDER -->', '')
        remainder = remainder.replace("<!-- MD_TABLE_PLACEHOLDER -->", "")
        
        return remainder.strip(), md_tables, html_tables

    def _parse_markdown_table_to_data(self, table_md_str: str) -> Tuple[List[str], List[List[str]]]:
        lines = [line.strip() for line in table_md_str.strip().split('\n') if line.strip()]
        if not lines: return [], []
        header_str = ""
        data_lines_str = []
        header_idx = -1
        separator_idx = -1

        for i, line in enumerate(lines):
            if re.match(r'^[\|\s]*:?-+:?(\s*\|\s*:?-+:?)*[\|\s]*$', line) and line.count('|') >= 2:
                if i > 0 and lines[i-1].count('|') >= 1:
                    separator_idx = i
                    header_idx = i-1
                    break
        
        if header_idx != -1 and separator_idx != -1:
            header_str = lines[header_idx]
            data_lines_str = [
                line for line in lines[separator_idx+1:] 
                if line.count('|') >= 1 and not (re.match(r'^[\|\s]*:?-+:?(\s*\|\s*:?-+:?)*[\|\s]*$', line) and line.count('|') >= 2)
            ]
        elif lines and lines[0].count('|') >=1:
             header_str = lines[0]
             data_lines_str = [
                 line for line in lines[1:] 
                 if line.count('|') >= 1 and not (re.match(r'^[\|\s]*:?-+:?(\s*\|\s*:?-+:?)*[\|\s]*$', line) and line.count('|') >= 2)
            ]
        else:
            data_rows_no_header = []
            for line in lines:
                if line.count('|') >=1 and not (re.match(r'^[\|\s]*:?-+:?(\s*\|\s*:?-+:?)*[\|\s]*$', line) and line.count('|') >= 2):
                    data_rows_no_header.append([cell.strip() for cell in line.strip('|').split('|')])
            return [], data_rows_no_header

        headers = [h.strip() for h in header_str.strip('|').split('|')] if header_str else []
        data_rows = []
        for data_line in data_lines_str:
            data_rows.append([cell.strip() for cell in data_line.strip('|').split('|')])
        
        if headers:
            aligned_data_rows = []
            num_headers = len(headers)
            for row in data_rows:
                aligned_row = row[:num_headers] + [''] * (num_headers - len(row))
                aligned_data_rows.append(aligned_row)
            data_rows = aligned_data_rows
            
        return headers, data_rows

    def _format_table_data_to_text(self, headers: List[str], data_rows: List[List[str]], context_title: Optional[str] = None) -> str:
        if not headers and not data_rows:
            return "（一个空的或无法解析的表格）"
        result_parts = []
        table_context_prefix = f"关于“{context_title}”的" if context_title else ""
        result_parts.append(f"{table_context_prefix}表格信息如下：")

        if headers:
            header_descs = [f"第{self.chinese_numbers[i] if i < len(self.chinese_numbers) else i+1}列的表头是“{h}”" 
                            for i, h in enumerate(headers) if h.strip()]
            if header_descs:
                result_parts.append("表头（列名）包括：" + "；".join(header_descs) + "。")
            else:
                result_parts.append("该表格的表头为空或所有列名均为空。")
        else:
            result_parts.append("该表格没有检测到明确的表头。")

        if data_rows:
            result_parts.append("\n数据内容：")
            for row_idx, row in enumerate(data_rows):
                row_label = f"第{self.chinese_numbers[row_idx] if row_idx < len(self.chinese_numbers) else row_idx+1}行数据显示"
                cell_descs = []
                num_cols_to_iterate = len(headers) if headers else len(row)

                for col_idx in range(num_cols_to_iterate):
                    cell_value = row[col_idx].strip() if col_idx < len(row) else ""
                    if cell_value:
                        col_name_part = ""
                        if headers and col_idx < len(headers) and headers[col_idx].strip():
                            col_name_part = f"“{headers[col_idx].strip()}”"
                        else:
                            col_name_part = f"第{self.chinese_numbers[col_idx] if col_idx < len(self.chinese_numbers) else col_idx+1}列"
                        cell_descs.append(f"{col_name_part}的值是“{cell_value}”")
                
                if cell_descs:
                    result_parts.append(f"{row_label}：" + "；".join(cell_descs) + "。")
                else:
                    result_parts.append(f"{row_label}该行为空或所有单元格值处理后为空。")
        else:
            result_parts.append("该表格没有数据行。")
        return "\n".join(result_parts)

    def _process_markdown_table(self, table_md_str: str, section_title: str) -> None:
        headers, data_rows = self._parse_markdown_table_to_data(table_md_str)
        num_cols = len(headers) if headers else (len(data_rows[0]) if data_rows else 0)
        metadata = {
            "section_title": section_title, "type": "table", "table_format": "markdown",
            "is_converted_format": self.convert_table_to_text,
            "rows": len(data_rows), "columns": num_cols, "headers": headers
        }
        table_content_for_chunk = self._format_table_data_to_text(headers, data_rows, section_title) if self.convert_table_to_text else table_md_str
        self._add_chunk(table_content_for_chunk, metadata)

    def _process_html_table_str(self, html_table_str: str, section_title: str) -> None:
        parsed_html_table_data = self._process_table_spans(html_table_str)
        headers, data_rows = (parsed_html_table_data[0], parsed_html_table_data[1:]) if parsed_html_table_data and len(parsed_html_table_data) > 0 else ([], [])
        num_cols = len(headers) if headers else (len(parsed_html_table_data[0]) if parsed_html_table_data and parsed_html_table_data[0] else 0)
        metadata = {
            "section_title": section_title, "type": "table", "table_format": "html",
            "is_converted_format": self.convert_table_to_text,
            "rows": len(data_rows), "columns": num_cols, "headers": headers
        }
        table_content_for_chunk = self._format_table_data_to_text(headers, data_rows, section_title) if self.convert_table_to_text else self._convert_data_to_markdown_table(headers, data_rows)
        self._add_chunk(table_content_for_chunk, metadata)

    def _convert_data_to_markdown_table(self, headers: List[str], data_rows: List[List[str]]) -> str:
        if not headers and not data_rows: return ""
        md_lines = []
        effective_headers = [str(h) for h in headers] if headers else []
        if effective_headers:
            md_lines.append("| " + " | ".join(effective_headers) + " |")
            md_lines.append("|" + "|".join(["---"] * len(effective_headers)) + "|")
        for row in data_rows:
            num_cols = len(effective_headers) if effective_headers else len(row)
            str_row = [str(cell) for cell in row]
            aligned_row = str_row[:num_cols] + [''] * (num_cols - len(str_row))
            md_lines.append("| " + " | ".join(aligned_row) + " |")
        return "\n".join(md_lines)

    def _update_hierarchy_headers(self, hierarchy_headers: List[str], new_header: str) -> List[str]:
        def count_leading_hashes(header: str) -> int:
            print(f"DEBUG count_leading_hashes: header='{header}', type={type(header)}")
            if not header: return 0
            match = re.match(r"^(#+)", header)
            if match:
                print(f"DEBUG count_leading_hashes: match found, group(1)='{match.group(1)}'")
                return len(match.group(1))
            else:
                print("DEBUG count_leading_hashes: no match found")
                return 0
        def compare_header(h1: str, h2: str) -> int:
            c1, c2 = count_leading_hashes(h1), count_leading_hashes(h2)
            if c1 == c2: return 0; return -1 if c1 > c2 else 1
        result = hierarchy_headers.copy()
        while result and compare_header(new_header, result[-1]) >= 0: result.pop()
        if new_header.replace("#", "").strip(): result.append(new_header)
        return result

    def _markdown_to_sections(self, markdown_text: str) -> List[Tuple[List[str], str]]:
        sections: List[Tuple[List[str], str]] = []
        lines = markdown_text.split("\n")
        hierarchy_headers: List[str] = []
        current_text_lines: List[str] = []
        code_block_flag = False

        for line in lines:
            if line.startswith("```"):
                code_block_flag = not code_block_flag
                current_text_lines.append(line)
                continue
            if code_block_flag:
                current_text_lines.append(line)
                continue

            header_match = re.match(r"^#+\s", line)
            if header_match:
                current_content_str = "\n".join(current_text_lines).strip()
                if current_content_str:
                    sections.append((copy.deepcopy(hierarchy_headers), current_content_str))
                current_text_lines = []
                hierarchy_headers = self._update_hierarchy_headers(hierarchy_headers, line.strip())
            else:
                current_text_lines.append(line)
        
        last_content_str = "\n".join(current_text_lines).strip()
        if last_content_str:
            sections.append((copy.deepcopy(hierarchy_headers), last_content_str))
        
        if not sections and markdown_text.strip():
             sections.append(([], markdown_text.strip()))
        return sections

    def _extract_section_title(self, hierarchy_headers: List[str]) -> str:
        if not hierarchy_headers: return "引言"
        titles = [re.sub(r'^#+\s+', '', h).strip() for h in hierarchy_headers[-self.contain_closest_title_levels:] if re.sub(r'^#+\s+', '', h).strip()]
        return " > ".join(titles) if titles else "未命名章节"

    def _split_text_with_regex(self, text: str, separator: str, keep_separator: bool) -> List[str]:
        if not separator: return list(text)
        escaped_separator = re.escape(separator)
        if keep_separator:
            _splits = re.split(f"({escaped_separator})", text)
            if not _splits: return []
            result = []
            i = 0
            while i < len(_splits):
                current_part = _splits[i]
                if i + 1 < len(_splits) and _splits[i+1] == separator:
                    current_part += _splits[i+1]
                    i += 1
                if current_part:
                    result.append(current_part)
                i += 1
            return [s for s in result if s]
        else:
            splits = re.split(escaped_separator, text)
            return [s for s in splits if s]

    def _langchain_style_merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc_parts: List[str] = []
        current_length = 0
        
        for s_val in splits:
            if not self.allow_cross_header_content and re.match(r"^#+\s", s_val) and current_doc_parts:
                doc_text = "".join(current_doc_parts).strip()
                if doc_text: docs.append(doc_text)
                current_doc_parts = [s_val]
                current_length = len(s_val)
                continue

            if current_length + len(s_val) <= self.max_chunk_size:
                current_doc_parts.append(s_val)
                current_length += len(s_val)
            else:
                if current_doc_parts:
                    doc_text = "".join(current_doc_parts).strip()
                    if doc_text: docs.append(doc_text)
                
                if docs and self.chunk_overlap > 0:
                    overlap_content = docs[-1][-self.chunk_overlap:]
                    if s_val.startswith(overlap_content):
                        current_doc_parts = [s_val]
                    else:
                        current_doc_parts = [overlap_content, s_val]
                else:
                    current_doc_parts = [s_val]
                current_length = sum(len(p) for p in current_doc_parts)
        
        if current_doc_parts:
            doc_text = "".join(current_doc_parts).strip()
            if doc_text: docs.append(doc_text)
        return docs
        
    def _recursive_split_no_headers(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        if len(text) <= self.max_chunk_size and not separators:
             if text.strip(): final_chunks.append(text.strip())
             return final_chunks
        
        current_separator = ""
        next_level_separators = []
        
        for i, sep in enumerate(separators):
            if sep == "":
                current_separator = ""
                next_level_separators = []
                break
            if re.search(re.escape(sep), text):
                current_separator = sep
                next_level_separators = separators[i+1:]
                break
        
        splits = self._split_text_with_regex(text, current_separator, keep_separator=True)
        good_splits: List[str] = []
        for s_val in splits:
            if len(s_val) <= self.max_chunk_size:
                good_splits.append(s_val)
            else:
                if good_splits:
                    merged = self._langchain_style_merge_splits(good_splits, current_separator if current_separator else "")
                    final_chunks.extend(merged)
                    good_splits = []
                
                if next_level_separators:
                    recursive_chunks = self._recursive_split_no_headers(s_val, next_level_separators)
                    final_chunks.extend(recursive_chunks)
                else:
                    if s_val.strip():
                        start = 0
                        while start < len(s_val):
                            end = start + self.max_chunk_size
                            chunk_part = s_val[start:end]
                            final_chunks.append(chunk_part)
                            start_offset = self.max_chunk_size - self.chunk_overlap
                            if start_offset <= 0 : start_offset = self.max_chunk_size // 2
                            start += start_offset
                            if start >= len(s_val) - self.chunk_overlap / 2 and start < len(s_val):
                                if len(s_val) - start > 0:
                                   final_chunks.append(s_val[start:])
                                break
        
        if good_splits:
            merged = self._langchain_style_merge_splits(good_splits, current_separator if current_separator else "")
            final_chunks.extend(merged)
        return [chunk for chunk in final_chunks if chunk.strip()]

    def _recursive_split_text(self, text: str, separators: Optional[List[str]] = None) -> List[str]:
        current_separators = separators if separators is not None else self.separators
        if not self.allow_cross_header_content:
            header_pattern = r"(?:^|\n)(?=#+\s)"
            sub_sections = re.split(header_pattern, text)
            final_chunks = []
            for sub_section in sub_sections:
                if sub_section.strip():
                    chunks_from_subsection = self._recursive_split_no_headers(sub_section.strip(), current_separators)
                    final_chunks.extend(chunks_from_subsection)
            return final_chunks
        else:
            return self._recursive_split_no_headers(text, current_separators)

    def _process_section_content(self, content: str, section_title: str) -> None:
        section_remainder, section_md_tables, section_html_tables = self._extract_tables_and_remainder(content)
        
        for table_md_str in section_md_tables:
            if table_md_str.strip():
                self._process_markdown_table(table_md_str, section_title)
        
        for html_table_str_item in section_html_tables:
            if html_table_str_item.strip():
                self._process_html_table_str(html_table_str_item, section_title)
        
        text_content = section_remainder.strip()
        if not text_content:
            return
        
        chunks_list = self._recursive_split_text(text_content)
        for chunk_text in chunks_list:
            if chunk_text.strip():
                metadata = {
                    "section_title": section_title,
                    "type": "text",
                    "char_count": len(chunk_text)
                }
                self._add_chunk(chunk_text, metadata)

    def _clean_cell_content(self, cell_content: str) -> str:
        content = re.sub(r'<[^>]+>', '', cell_content)
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    def _process_table_spans(self, html_table_str: str) -> List[List[str]]:
        html_table_str = html_table_str.lower()
        html_table_str = re.sub(r'\s*\n\s*', '', html_table_str)
        try:
            row_matches = re.findall(r'<tr[^>]*>(.*?)</tr>', html_table_str, re.IGNORECASE)
            if not row_matches: return []
            grid: List[List[Optional[str]]] = []
            for r_idx, row_content in enumerate(row_matches):
                while len(grid) <= r_idx: grid.append([])
                cell_iter = re.finditer(r'<t[dh]([^>]*)>(.*?)</t[dh]>', row_content, re.IGNORECASE)
                current_col_idx_in_grid = 0
                for cell_match in cell_iter:
                    attrs_str, cell_inner_html = cell_match.group(1), cell_match.group(2)
                    content = self._clean_cell_content(cell_inner_html)
                    colspan = int(m.group(1)) if (m := re.search(r'colspan\s*=\s*["\']?(\d+)["\']?', attrs_str)) else 1
                    rowspan = int(m.group(1)) if (m := re.search(r'rowspan\s*=\s*["\']?(\d+)["\']?', attrs_str)) else 1
                    while current_col_idx_in_grid < len(grid[r_idx]) and grid[r_idx][current_col_idx_in_grid] is not None:
                        current_col_idx_in_grid += 1
                    for i_row in range(rowspan):
                        target_row_idx = r_idx + i_row
                        while len(grid) <= target_row_idx: grid.append([])
                        for j_col in range(colspan):
                            target_col_idx = current_col_idx_in_grid + j_col
                            while len(grid[target_row_idx]) <= target_col_idx: grid[target_row_idx].append(None)
                            if grid[target_row_idx][target_col_idx] is None: grid[target_row_idx][target_col_idx] = content
                    current_col_idx_in_grid += colspan
            if not grid: return []
            max_cols = max(len(r) for r in grid) if grid else 0
            final_table_data = []
            for r_data in grid:
                row_list = [(cell if cell is not None else "") for cell in r_data]
                row_list.extend([""] * (max_cols - len(row_list)))
                final_table_data.append(row_list)
            return final_table_data
        except Exception as e:
            print(f"处理HTML表格跨行跨列时出错: {str(e)}")
            return []

    def save_to_json(self, output_path: str, ensure_ascii: bool = False, indent: int = 2) -> None:
        try:
            with open(output_path, 'w', encoding=self.encoding) as f:
                json.dump(self.chunks, f, ensure_ascii=ensure_ascii, indent=indent)
            print(f"结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存JSON文件时出错: {str(e)}")

    def get_chunks(self) -> List[Dict]:
        return self.chunks
