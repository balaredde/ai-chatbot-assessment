from model_loader import ModelLoader
from chat_memory import ChatMemory
import sys

class ChatInterface:
    """
    Command-line interface for the chatbot with conversation management.
    CPU-optimized version.
    """
    
    def __init__(self, window_size=5):
        """
        Initialize the chatbot interface.
        
        Args:
            window_size (int): Number of conversation turns to remember
        """
        print("=" * 70)
        print(" " * 20 + "AI CHATBOT INTERFACE")
        print("=" * 70)
        print("\nInitializing components...")
        
        # Initialize model loader
        try:
            self.model = ModelLoader()
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            sys.exit(1)
        
        # Initialize memory buffer
        self.memory = ChatMemory(window_size=window_size)
        
        # Display info
        device_info = self.model.get_device_info()
        print(f"\n{'─' * 70}")
        print(f"✓ Running on: {device_info['device']}")
        print(f"✓ Memory Window: {window_size} conversation turns")
        print(f"{'─' * 70}")
        
        # Display commands
        self._display_help()
    
    def _display_help(self):
        """Display available commands."""
        print("\nAvailable Commands:")
        print("  /exit  - Exit the chatbot")
        print("  /clear - Clear conversation memory")
        print("  /info  - Display memory statistics")
        print("  /help  - Show this help message")
        print(f"\n{'=' * 70}\n")
    
    def run(self):
        """Main CLI loop for continuous interaction."""
        print("Chatbot is ready! Start chatting...\n")
        
        while True:
            try:
                # Get user input
                user_input = input("User: ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() == "/exit":
                    print("\n" + "=" * 70)
                    print(" " * 25 + "GOODBYE!")
                    print("=" * 70)
                    print("Exiting chatbot. Thanks for chatting!\n")
                    break
                
                if user_input.lower() == "/clear":
                    self.memory.clear_memory()
                    print("✓ Conversation memory cleared.\n")
                    continue
                
                if user_input.lower() == "/info":
                    self._display_info()
                    continue
                
                if user_input.lower() == "/help":
                    self._display_help()
                    continue
                
                # Process normal chat message
                self._handle_user_message(user_input)
                
            except KeyboardInterrupt:
                print("\n\n" + "=" * 70)
                print(" " * 20 + "INTERRUPTED BY USER")
                print("=" * 70)
                print("Exiting chatbot. Goodbye!\n")
                break
                
            except Exception as e:
                print(f"\n✗ Unexpected error: {e}")
                print("Please try again or type /exit to quit.\n")
    
    def _handle_user_message(self, user_input):
        """
        Process user message and generate bot response.
        
        Args:
            user_input (str): The user's message
        """
        # Add user message to memory
        self.memory.add_user_message(user_input)
        
        # Get conversation history with context
        messages = self.memory.get_conversation_history()
        
        # Generate response
        print("Bot: ", end="", flush=True)
        
        try:
            full_response = self.model.generate_response(messages)
            
            # Extract only the assistant's latest response
            assistant_response = self._extract_assistant_response(full_response)
            
            # Limit response length
            assistant_response = self._limit_response_length(assistant_response)
            
            # Add assistant response to memory
            self.memory.add_assistant_message(assistant_response)
            
            # Display response
            print(assistant_response + "\n")
            
        except Exception as e:
            error_msg = "Sorry, I encountered an error generating a response."
            print(error_msg + "\n")
            print(f"[Debug] Error: {e}\n")
    
    def _extract_assistant_response(self, full_response):
        """Extract the assistant's response from full generated text."""
        # Handle list format
        if isinstance(full_response, list):
            full_response = full_response[-1].get("content", full_response[-1])
        
        # Handle dict format
        if isinstance(full_response, dict):
            full_response = full_response.get("content", str(full_response))
        
        # Convert to string
        full_response = str(full_response)
        
        # Remove chat template markers
        if "<|assistant|>" in full_response:
            parts = full_response.split("<|assistant|>")
            if len(parts) > 1:
                response = parts[-1].strip()
                response = response.split("<|")[0].strip()
                return response
        
        # Handle other formats
        if "Assistant:" in full_response:
            parts = full_response.split("Assistant:")
            return parts[-1].strip()
        
        if "User:" in full_response:
            parts = full_response.split("User:")
            last_part = parts[-1]
            if "Bot:" in last_part or "Assistant:" in last_part:
                response = last_part.split("Bot:")[-1] if "Bot:" in last_part else last_part.split("Assistant:")[-1]
                return response.strip()
        
        return full_response.strip()
    
    def _limit_response_length(self, response, max_sentences=3, max_chars=300):
        """Limit response length to prevent verbosity."""
        # Remove extra whitespace
        response = " ".join(response.split())
        
        # Limit by sentences
        sentences = response.split('. ')
        if len(sentences) > max_sentences:
            response = '. '.join(sentences[:max_sentences])
            if not response.endswith('.'):
                response += '.'
        
        # Limit by character count
        if len(response) > max_chars:
            response = response[:max_chars].rsplit(' ', 1)[0] + '...'
        
        return response
    
    def _display_info(self):
        """Display memory usage statistics."""
        memory_summary = self.memory.get_conversation_summary()
        
        print("\n" + "=" * 70)
        print(" " * 25 + "SYSTEM INFO")
        print("=" * 70)
        
        print("\n💾 Conversation Memory:")
        print(f"  • Total Messages: {memory_summary['total_messages']}")
        print(f"  • User Messages: {memory_summary['user_messages']}")
        print(f"  • Assistant Messages: {memory_summary['assistant_messages']}")
        print(f"  • Conversation Turns: {memory_summary['conversation_turns']}")
        print(f"  • Memory Usage: {memory_summary['memory_usage_percent']:.1f}%")
        print(f"  • Memory Capacity: {memory_summary['memory_capacity']} messages")
        
        print("=" * 70 + "\n")


if __name__ == "__main__":
    """Entry point for the chatbot application."""
    try:
        chatbot = ChatInterface(window_size=5)
        chatbot.run()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        print("Unable to start chatbot. Please check your installation.\n")
        sys.exit(1)
