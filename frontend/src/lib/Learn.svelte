<script lang="ts">
	import { goto } from '$app/navigation';

	const categories = [
		'Abdomen – General',
		'Abdomen – Hernia',
		'Abdomen – Spleen',
		'Alimentary Tract – Stomach',
		'Alimentary Tract – Small Intestine',
		'Alimentary Tract – Large Intestine',
		'Endocrine',
		'Skin and Soft Tissue',
		'Surgical Critical Care',
		'Trauma',
		'Thoracic Surgery',
		'Plastic Surgery'
	];

	function handleCategoryClick(category: string) {
		const path = `/learn${category.replace(/\s+/g, '')}`;
		goto(path);
	}
</script>

<div class="w-full h-full p-6">
	<h1 class="text-3xl font-bold text-center mb-8 text-gray-800">Learning Categories</h1>
	
	<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-6xl mx-auto">
		{#each categories as category}
			<button
				class="border-4 border-black py-4 px-6 text-center min-h-[80px] flex items-center justify-center font-semibold"
				on:click={() => handleCategoryClick(category)}
			>
				{category}
			</button>
		{/each}
	</div>
</div> 
