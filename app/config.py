from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="GROUNDED_SQL_AGENT_DEBUG")

    # LLM Provider
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
    )

    # Default model
    default_ollama_model: str = Field(
        default="llama3.1",
        alias="DEFAULT_OLLAMA_MODEL",
    )

    # Role-specific models
    interpret_model: str = Field(default="llama3.1", alias="INTERPRET_MODEL")
    mapping_model: str = Field(default="llama3.1", alias="MAPPING_MODEL")
    clarification_model: str = Field(default="llama3.1", alias="CLARIFICATION_MODEL")
    sql_model: str = Field(default="llama3.1", alias="SQL_MODEL")
    answer_model: str = Field(default="llama3.1", alias="ANSWER_MODEL")

    # Temperatures
    interpret_temperature: float = Field(default=0.0, alias="INTERPRET_TEMPERATURE")
    mapping_temperature: float = Field(default=0.0, alias="MAPPING_TEMPERATURE")
    clarification_temperature: float = Field(default=0.3, alias="CLARIFICATION_TEMPERATURE")
    sql_temperature: float = Field(default=0.0, alias="SQL_TEMPERATURE")
    answer_temperature: float = Field(default=0.2, alias="ANSWER_TEMPERATURE")

    # Clarification loop
    max_clarification_count: int = Field(default=5, alias="MAX_CLARIFICATION_COUNT")

    # BigQuery
    gcp_project_id: str = Field(
        default="your-gcp-project-id",
        alias="GCP_PROJECT_ID",
    )
    bigquery_dataset_id: str = Field(
        default="your_bigquery_dataset",
        alias="BIGQUERY_DATASET_ID",
    )
    bigquery_table_id: str = Field(
        default="your_bigquery_table",
        alias="BIGQUERY_TABLE_ID",
    )
    bigquery_location: str = Field(
        default="YOUR_BIGQUERY_LOCATION",
        alias="BIGQUERY_LOCATION",
    )
    bigquery_logical_table_name: str = Field(
        default="sensor_measurements",
        alias="BIGQUERY_LOGICAL_TABLE_NAME",
    )

    # LangSmith
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT",
    )
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="sql-agent-dev", alias="LANGSMITH_PROJECT")

    @property
    def bigquery_physical_table_name(self) -> str:
        return (
            f"`{self.gcp_project_id}."
            f"{self.bigquery_dataset_id}."
            f"{self.bigquery_table_id}`"
        )


settings = Settings()
