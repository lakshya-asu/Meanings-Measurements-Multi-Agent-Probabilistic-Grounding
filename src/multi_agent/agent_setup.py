from src.multi_agent.agents.orchestrator_agent import OrchestratorAgent
from src.multi_agent.agents.grounding_agent import GroundingAgent
from src.multi_agent.agents.spatial_agent import SpatialAgent
from src.multi_agent.agents.verifier_agent import VerifierAgent
from src.multi_agent.agents.logical_agent import LogicalAgent
from src.multi_agent.agents.qa_agent import QaAgent

class AgentFactory:
    @staticmethod
    def create_agent(role: str, provider: str = "gemini", **kwargs):
        role = role.lower()
        provider = provider.lower()
        
        if role == "orchestrator":
            if provider == "openai":
                from src.multi_agent.agents.openai_orchestrator_agent import OpenAIOrchestratorAgent
                return OpenAIOrchestratorAgent(**kwargs)
            elif provider == "claude":
                from src.multi_agent.agents.claude_orchestrator_agent import ClaudeOrchestratorAgent
                return ClaudeOrchestratorAgent(**kwargs)
            elif provider in ["alibaba", "qwen"]:
                from src.multi_agent.agents.alibaba_orchestrator_agent import AlibabaOrchestratorAgent
                return AlibabaOrchestratorAgent(**kwargs)
            return OrchestratorAgent(**kwargs)
            
        elif role == "logical":
            if provider == "openai":
                from src.multi_agent.agents.openai_logical_agent import OpenAILogicalAgent
                return OpenAILogicalAgent(**kwargs)
            elif provider == "claude":
                from src.multi_agent.agents.claude_logical_agent import ClaudeLogicalAgent
                return ClaudeLogicalAgent(**kwargs)
            elif provider in ["alibaba", "qwen"]:
                from src.multi_agent.agents.alibaba_logical_agent import AlibabaLogicalAgent
                return AlibabaLogicalAgent(**kwargs)
            return LogicalAgent(**kwargs)
            
        elif role == "grounding":
            if provider == "openai":
                from src.multi_agent.agents.openai_grounding_agent import OpenAIGroundingAgent
                return OpenAIGroundingAgent(**kwargs)
            elif provider == "claude":
                from src.multi_agent.agents.claude_grounding_agent import ClaudeGroundingAgent
                return ClaudeGroundingAgent(**kwargs)
            elif provider in ["alibaba", "qwen"]:
                from src.multi_agent.agents.alibaba_grounding_agent import AlibabaGroundingAgent
                return AlibabaGroundingAgent(**kwargs)
            return GroundingAgent(**kwargs)
            
        elif role == "spatial":
            if provider == "openai":
                from src.multi_agent.agents.openai_spatial_agent import OpenAISpatialAgent
                return OpenAISpatialAgent(**kwargs)
            elif provider == "claude":
                from src.multi_agent.agents.claude_spatial_agent import ClaudeSpatialAgent
                return ClaudeSpatialAgent(**kwargs)
            elif provider in ["alibaba", "qwen"]:
                from src.multi_agent.agents.alibaba_spatial_agent import AlibabaSpatialAgent
                return AlibabaSpatialAgent(**kwargs)
            return SpatialAgent(**kwargs)
            
        elif role == "verifier":
            if provider == "openai":
                from src.multi_agent.agents.openai_verifier_agent import OpenAIVerifierAgent
                return OpenAIVerifierAgent(**kwargs)
            elif provider == "claude":
                from src.multi_agent.agents.claude_verifier_agent import ClaudeVerifierAgent
                return ClaudeVerifierAgent(**kwargs)
            elif provider in ["alibaba", "qwen"]:
                from src.multi_agent.agents.alibaba_verifier_agent import AlibabaVerifierAgent
                return AlibabaVerifierAgent(**kwargs)
            return VerifierAgent(**kwargs)
            
        elif role == "qa":
            if provider == "openai":
                from src.multi_agent.agents.openai_qa_agent import OpenAIQaAgent
                return OpenAIQaAgent(**kwargs)
            elif provider == "claude":
                from src.multi_agent.agents.claude_qa_agent import ClaudeQaAgent
                return ClaudeQaAgent(**kwargs)
            elif provider in ["alibaba", "qwen"]:
                from src.multi_agent.agents.alibaba_qa_agent import AlibabaQaAgent
                return AlibabaQaAgent(**kwargs)
            return QaAgent(**kwargs)
            
        else:
            raise ValueError(f"Unknown agent role: {role}")
