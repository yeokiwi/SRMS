import os
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import LLMConfig
from crawl4ai.content_filter_strategy import LLMContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from dotenv import load_dotenv
from datetime import date
from crawl4ai.deep_crawling import BestFirstCrawlingStrategy

# Get today's date object
today = date.today()

# Format the date as a string
formatted_date = today.strftime("%d %b, %Y") 

async def test_llm_filter(url=None, output_filename=None, directory=None,max_depth_no=1):
    # Create an HTML source that needs intelligent filtering
    if url is None:
        url = "https://www.iras.gov.sg/taxes/goods-services-tax-(gst)"
    if output_filename is None:
        output_filename = "filtered_content_deepcrawl.md"
    
    # Handle directory parameter
    if directory:
        os.makedirs(directory, exist_ok=True)
        output_path = os.path.join(directory, output_filename)
    else:
        output_path = output_filename
    
    browser_config = BrowserConfig(
        headless=True,
        enable_stealth=True,
        user_agent_mode="random",
        verbose=True
    )
    
    # Configure deep crawl strategy: max_depth=1 means initial page + one level down
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=max_depth_no,
        include_external=False  # Only follow links within the same domain
    )
    
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        deep_crawl_strategy=deep_crawl_strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        #excluded_tags=["nav", "ul", "li", "a"],
        verbose=True
    )
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize LLM filter with focused instruction
    filter = LLMContentFilter(
        llm_config=LLMConfig(provider="deepseek/deepseek-chat", api_token=os.getenv('DEEPSEEK_API_KEY')),
        instruction=f"""
        You are a helpful summary assistant. Only summarize the web page with information that are last updated one month from {formatted_date}. 
        Indicate whether the page is last updated one month from {formatted_date}.
        Indicate the page update status at the top of the content.
        """,
        verbose=True
    )
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Perform deep crawl (max_depth=1: initial page + one level down)
        print(f"Starting deep crawl of {url} (max_depth=1)...")
        results = await crawler.arun(url, config=run_config)
        
        print(f"\n[SUCCESS] Crawled {len(results)} pages total")
        
        all_filtered_content = []
        
        # Process each crawled page
        for i, result in enumerate(results):
            print(f"\n--- Processing page {i+1}/{len(results)} ---")
            print(f"URL: {result.url}")
            print(f"Depth: {result.metadata.get('depth', 0)}")
            
            # Save crawled web page content if directory is specified
            if directory:
                # Create a safe filename from URL and index
                import re
                safe_filename = f"{url}page_{i+1}_depth_{result.metadata.get('depth', 0)}"
                # Remove or replace problematic characters
                safe_filename = re.sub(r'[^\w\-_\. ]', '_', safe_filename)
                
                # Save HTML content
                html_filename = os.path.join(directory, f"{safe_filename}.html")
                html_content = result.html if hasattr(result, 'html') and result.html else ""
                with open(html_filename, "w", encoding="utf-8") as f:
                    f.write(html_content)
                print(f"[SAVED] HTML to: {html_filename}")
                
                # Save cleaned HTML if available
                if hasattr(result, 'cleaned_html') and result.cleaned_html:
                    cleaned_html_filename = os.path.join(directory, f"{safe_filename}_cleaned.html")
                    with open(cleaned_html_filename, "w", encoding="utf-8") as f:
                        f.write(result.cleaned_html)
                    print(f"[SAVED] cleaned HTML to: {cleaned_html_filename}")
                
                # Save markdown if available
                if hasattr(result, 'markdown') and result.markdown:
                    markdown_filename = os.path.join(directory, f"{safe_filename}.md")
                    # Handle both string and object with raw_markdown attribute
                    if isinstance(result.markdown, str):
                        markdown_content = result.markdown
                    elif hasattr(result.markdown, 'raw_markdown'):
                        markdown_content = result.markdown.raw_markdown
                    else:
                        markdown_content = str(result.markdown)
                    
                    with open(markdown_filename, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    print(f"[SAVED] markdown to: {markdown_filename}")
            
            # Apply LLM filtering to this page's content
            filtered_content = filter.filter_content(result.cleaned_html)
            
            if filtered_content:
                # Add metadata about which page this came from
                page_info = f"\n\n## Page {i+1}: {result.url} (Depth: {result.metadata.get('depth', 0)})"
                all_filtered_content.append(page_info)
                all_filtered_content.extend(filtered_content)
                
                print(f"Filtered content length: {len(filtered_content)} chunks")
                if filtered_content:
                    print(f"First 300 chars: {filtered_content[0][:300]}...")
            else:
                print("No filtered content returned for this page")
        
        # Show combined results
        print("\n" + "="*60)
        print("COMBINED RESULTS")
        print("="*60)
        print(f"Total filtered content chunks: {len(all_filtered_content)}")
        
        if all_filtered_content:
            # Save combined markdown version
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(all_filtered_content))
            print(f"\n[SAVED] Saved combined filtered content to '{output_path}'")
            
            # Show sample of combined content
            print("\nSample of combined content (first 800 chars):")
            combined_text = "\n".join(all_filtered_content)
            print(combined_text[:800] + "..." if len(combined_text) > 800 else combined_text)
        else:
            print("\n[WARNING] No filtered content was generated from any page")
        
        # Show token usage
        print("\n" + "="*60)
        print("TOKEN USAGE SUMMARY")
        print("="*60)
        filter.show_usage()
        
        # Print crawl statistics
        print("\n" + "="*60)
        print("CRAWL STATISTICS")
        print("="*60)
        
        # Group results by depth
        pages_by_depth = {}
        for result in results:
            depth = result.metadata.get("depth", 0)
            if depth not in pages_by_depth:
                pages_by_depth[depth] = []
            pages_by_depth[depth].append(result.url)
        
        for depth, urls in sorted(pages_by_depth.items()):
            print(f"Depth {depth}: {len(urls)} pages")
            for url in urls[:3]:  # Show first 3 URLs for each depth
                print(f"  -> {url}")
            if len(urls) > 3:
                print(f"  ... and {len(urls) - 3} more")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep crawl a website and apply LLM filtering")
    parser.add_argument("--url", type=str, help="URL to crawl (default: MOM workplace safety page)")
    parser.add_argument("--output", type=str, help="Output markdown filename (default: filtered_content_deepcrawl.md)")
    parser.add_argument("--directory", type=str, help="Directory to save crawled web pages and markdown file")
    
    args = parser.parse_args()
    
#    asyncio.run(test_llm_filter(url=args.url, output_filename=args.output, directory=args.directory))
#    asyncio.run(test_llm_filter("https://www.acra.gov.sg/", "acra.md", "acra"))
#    asyncio.run(test_llm_filter("https://www.imda.gov.sg/regulations-and-licensing-listing/telecommunications-act-1999", "imda.md", "imda"))
#    asyncio.run(test_llm_filter("https://www.acra.gov.sg/accountancy/accounting-standards/pronouncements/financial-reporting-standards/archives/2023-volume", "acra2.md", "acra2"))
#    asyncio.run(test_llm_filter("https://www.singstat.gov.sg/", "singstat.md", "singstat"))
#    asyncio.run(test_llm_filter("https://www.caas.gov.sg/legislation-regulations/legislation/air-navigation-act", "caas.md", "caas"))
#    asyncio.run(test_llm_filter("https://www.mom.gov.sg/employment-practices/employment-act", "mom1.md", "mom1"))
#    asyncio.run(test_llm_filter("https://www.mom.gov.sg/workplace-safety-and-health/work-injury-compensation", "mom2.md", "mom2"))
#    asyncio.run(test_llm_filter("https://www.profamilyleave.msf.gov.sg/", "msf.md", "msf"))
#    asyncio.run(test_llm_filter("https://www.cpf.gov.sg/member/", "cpf.md", "cpf"))
#    asyncio.run(test_llm_filter("https://www.mom.gov.sg/employment-practices/retirement", "mom3.md", "mom3"))
#    asyncio.run(test_llm_filter("https://www.mom.gov.sg/employment-practices/re-employment", "mom4.md", "mom4"))
#    asyncio.run(test_llm_filter("https://www.mtcr.info/en/mtcr-annex", "mtcr.md", "mtcr"))
#    asyncio.run(test_llm_filter("https://www.imda.gov.sg/regulations-and-licensing-listing/spectrum-management/frequency-allocation-and-assignment", "imda.md", "imda"))
#    asyncio.run(test_llm_filter("https://biosafety.moh.gov.sg/", "moh1.md", "moh1"))
#    asyncio.run(test_llm_filter("https://www.opcw.org/chemical-weapons-convention", "opcw.md", "opcw",0))
#    asyncio.run(test_llm_filter("http://www.gmac.sg/news/", "gmac.md", "gmac",1))
#    asyncio.run(test_llm_filter("https://hpp.moh.gov.sg/medical-acts-statutes/", "moh2.md", "moh2",1))
#    asyncio.run(test_llm_filter("https://www.hsa.gov.sg/poisons", "hsa1.md", "hsa1",1))
#    asyncio.run(test_llm_filter("https://www.hsa.gov.sg/controlled-drugs-psychotropic-substances/controlled-drugs/apply", "hsa2.md", "hsa2",0))
#    asyncio.run(test_llm_filter("https://www.hcsa.gov.sg/resources/", "hcsa.md", "hcsa",1))
    asyncio.run(test_llm_filter("https://www.mom.gov.sg/workplace-safety-and-health/workplace-safety-and-health-act", "mom5.md", "mom5",1))
