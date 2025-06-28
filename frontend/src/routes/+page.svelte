<script lang="ts">
	import Ask from '$lib/Ask.svelte';
	import Learn from '$lib/Learn.svelte';
	
	let page: 'ask' | 'learn' = $state('learn');
</script>

<main class="flex w-screen h-screen pt-20 justify-center">
	<div class="w-2/3 flex-col h-[80%] flex items-center border-black border-4 p-4 gap-4">
		<nav class="flex flex-row w-full gap-4 justify-center">
			<button class="border-4 border-black w-20 h-10" onclick={() => (page = 'ask')}>ask</button>
			<button class="border-4 border-black w-20 h-10" onclick={() => (page = 'learn')}>learn</button
			>
		</nav>
		{#if page == 'ask'}
			<Ask />
		{:else}
			<Learn />
		{/if}
	</div>
</main>
