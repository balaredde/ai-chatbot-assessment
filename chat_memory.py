from collections import deque

class ChatMemory:
    """
    Manages conversation history using a sliding window buffer.
    Implements a deque with maxlen to automatically maintain the last N turns.
    """
    
    def __init__(self, window_size=5):
        """
        Initialize chat memory with sliding window.
        
        Args:
            window_size (int): Maximum number of message pairs to keep in memory
        """
        self.window_size = window_size
        self.memory = deque(maxlen=window_size * 2)  # *2 for user+assistant pairs
        self.system_prompt = {
            "role": "system",
            "content": "You are a helpful AI assistant. Provide concise, accurate answers. If you don't know something, say so instead of guessing. Keep responses brief and factual."
        }
    
    def add_user_message(self, content):
        """
        Add user message to memory buffer.
        
        Args:
            content (str): User's message content
        """
        self.memory.append({"role": "user", "content": content})
    
    def add_assistant_message(self, content):
        """
        Add assistant message to memory buffer.
        
        Args:
            content (str): Assistant's response content
        """
        self.memory.append({"role": "assistant", "content": content})
    
    def get_conversation_history(self):
        """
        Get formatted conversation history for the model.
        
        Returns:
            list: List of message dictionaries with system prompt + memory
        """
        messages = [self.system_prompt]
        messages.extend(list(self.memory))
        return messages
    
    def clear_memory(self):
        """Clear all conversation history from buffer."""
        self.memory.clear()
    
    def get_memory_size(self):
        """
        Return current number of messages in memory.
        
        Returns:
            int: Number of messages (not pairs) in memory
        """
        return len(self.memory)
    
    def get_last_user_message(self):
        """
        Get the most recent user message.
        
        Returns:
            str or None: Last user message content or None if empty
        """
        for msg in reversed(self.memory):
            if msg["role"] == "user":
                return msg["content"]
        return None
    
    def get_conversation_summary(self):
        """
        Get a summary of the conversation state.
        
        Returns:
            dict: Summary including turn count and memory usage
        """
        user_msgs = sum(1 for msg in self.memory if msg["role"] == "user")
        assistant_msgs = sum(1 for msg in self.memory if msg["role"] == "assistant")
        
        return {
            "total_messages": len(self.memory),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "conversation_turns": min(user_msgs, assistant_msgs),
            "memory_capacity": self.window_size * 2,
            "memory_usage_percent": (len(self.memory) / (self.window_size * 2)) * 100
        }
