<script lang="ts">
	let message = $state('');
	
	function handleSubmit() {
		// Handle message submission here
		console.log('Message submitted:', message);
		message = ''; // Clear the input after submission
	}
</script>

<div class="w-full h-full flex flex-col gap-4">
	<div class="flex-1 border-4 border-black p-4">
		<textarea 
			bind:value={message}
			placeholder="Type your message here..."
			class="w-full h-full resize-none outline-none border-none bg-transparent"
			onkeydown={(e) => {
				if (e.key === 'Enter' && !e.shiftKey) {
					e.preventDefault();
					handleSubmit();
				}
			}}
		></textarea>
	</div>
	<div class="flex justify-end">
		<button 
			class="border-4 border-black w-20 h-10"
			onclick={handleSubmit}
		>
			Send
		</button>
	</div>
</div> 