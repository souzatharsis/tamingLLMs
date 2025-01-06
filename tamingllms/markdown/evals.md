(evals)=
# The Evals Gap
```{epigraph}
It doesn't matter how beautiful your theory is, <br>
it doesn't matter how smart you are. <br>
If it doesn't agree with experiment, it's wrong.

-- Richard Feynman
```
```{contents}
```

## Introduction

The advent of LLMs marks a pivotal shift in the landscape of software development, testing and verification. Unlike traditional software systems, where deterministic outputs are the norm, LLMs introduce a realm of non-deterministic and generative behaviors that challenge conventional software engineering paradigms. This shift is not merely a technical evolution but a fundamental transformation in how we conceive, build, and assess software products.

For those entrenched in traditional methodologies, the transition to LLM-driven systems may seem daunting. However, ignoring this change is not an option. The reliance on outdated testing frameworks that fail to account for the probabilistic nature of LLMs will inevitably lead to significant setbacks.

To overcome these challenges, it is imperative to embrace the complexities of LLMs with a proactive mindset. This involves developing robust evaluation frameworks up-front that incorporate the generative nature of LLM-based software development while fostering a culture of continuous change, learning and adaptation.


## Non-Deterministic Generative Machines

One of the most fundamental challenges when building products with LLMs is their generative and non-deterministic nature. Unlike traditional software systems where the same input reliably produces the same output, LLMs can generate novel text that may not exist in their training data, and produce different responses each time they're queried - even with identical prompts and input data. This behavior is both a strength and a significant engineering and product challenge.

When you ask an LLM the same question multiple times, you'll likely get different responses. This isn't a bug - it's a fundamental feature of how these models work. The "temperature" parameter, which controls the randomness of outputs, allows models to be creative and generate diverse responses. However, this same feature makes it difficult to build reliable, testable systems.

Consider a financial services company using LLMs to generate investment advice. The non-deterministic nature of these models means that:
- The same input data could yield different analysis conclusions
- Regulatory compliance becomes challenging to guarantee
- User trust may be affected by inconsistent responses
- Testing becomes exceedingly more complex compared to traditional software

The primary source of non-determinism in LLMs comes from their sampling strategies. During text generation, the model:
1. Calculates probability distributions for each next token
2. Samples from these distributions based on temperature settings
3. Uses techniques like nucleus sampling {cite}`holtzman2020curiouscaseneuraltext` or top-k sampling to balance creativity and coherence

In this simple experiment, we use an LLM to write a single-statement executive summary from an input financial filing. We observe that even a simple parameter like temperature can dramatically alter model behavior in ways that are difficult to systematically assess. At temperature 0.0, responses are consistent but potentially too rigid. At 1.0, outputs become more varied but less predictable. At 2.0, responses can be wildly different and often incoherent. This non-deterministic behavior makes traditional software testing approaches inadequate.


```python
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from openai import OpenAI
import pandas as pd
from typing import List

def generate_responses(
    model_name: str,
    prompt: str,
    temperatures: List[float],
    attempts: int = 3
) -> pd.DataFrame:
    """
    Generate multiple responses at different temperature settings
    to demonstrate non-deterministic behavior.
    """
    client = OpenAI()
    results = []
    
    for temp in temperatures:
        for attempt in range(attempts):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=50
            )
            
            results.append({
                'temperature': temp,
                'attempt': attempt + 1,
                'response': response.choices[0].message.content
            })

    # Display results grouped by temperature
    df_results = pd.DataFrame(results)
    for temp in temperatures:
        print(f"\nTemperature = {temp}")
        print("-" * 40)
        temp_responses = df_results[df_results['temperature'] == temp]
        for _, row in temp_responses.iterrows():
            print(f"Attempt {row['attempt']}: {row['response']}")
    
    return df_results
```


```python
MAX_LENGTH = 10000 # We limit the input length to avoid token issues
with open('../data/apple.txt', 'r') as file:
    sec_filing = file.read()
```


```python
sec_filing
```




    'UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nWashington, D.C. 20549\n \nFORM 10-K\n \n(Mark One)\n☒ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934\nFor the fiscal year ended September 28, 2024\nor\n☐ TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934\nFor the transition period from              to             .\nCommission File Number: 001-36743\n \ng66145g66i43.jpg\nApple Inc.\n(Exact name of Registrant as specified in its charter)\n \nCalifornia\t\t94-2404110\n(State or other jurisdiction\nof incorporation or organization)\n(I.R.S. Employer Identification No.)\nOne Apple Park Way\t\t\nCupertino, California\n95014\n(Address of principal executive offices)\t\t(Zip Code)\n \n(408) 996-1010\n(Registrant’s telephone number, including area code)\n \nSecurities registered pursuant to Section 12(b) of the Act:\nTitle of each class\tTrading symbol(s)\tName of each exchange on which registered\nCommon Stock, $0.00001 par value per share\nAAPL\tThe Nasdaq Stock Market LLC\n0.000% Notes due 2025\t—\tThe Nasdaq Stock Market LLC\n0.875% Notes due 2025\t—\tThe Nasdaq Stock Market LLC\n1.625% Notes due 2026\t—\tThe Nasdaq Stock Market LLC\n2.000% Notes due 2027\t—\tThe Nasdaq Stock Market LLC\n1.375% Notes due 2029\t—\tThe Nasdaq Stock Market LLC\n3.050% Notes due 2029\t—\tThe Nasdaq Stock Market LLC\n0.500% Notes due 2031\t—\tThe Nasdaq Stock Market LLC\n3.600% Notes due 2042\t—\tThe Nasdaq Stock Market LLC\n \nSecurities registered pursuant to Section 12(g) of the Act: None\n \nIndicate by check mark if the Registrant is a well-known seasoned issuer, as defined in Rule 405 of the Securities Act.\nYes  ☒     No  ☐\nIndicate by check mark if the Registrant is not required to file reports pursuant to Section 13 or Section 15(d) of the Act.\nYes  ☐     No  ☒\n\nIndicate by check mark whether the Registrant (1) has filed all reports required to be filed by Section 13 or 15(d) of the Securities Exchange Act of 1934 during the preceding 12 months (or for such shorter period that the Registrant was required to file such reports), and (2) has been subject to such filing requirements for the past 90 days.\nYes  ☒     No  ☐\nIndicate by check mark whether the Registrant has submitted electronically every Interactive Data File required to be submitted pursuant to Rule 405 of Regulation S-T (§232.405 of this chapter) during the preceding 12 months (or for such shorter period that the Registrant was required to submit such files).\nYes  ☒     No  ☐\nIndicate by check mark whether the Registrant is a large accelerated filer, an accelerated filer, a non-accelerated filer, a smaller reporting company, or an emerging growth company. See the definitions of “large accelerated filer,” “accelerated filer,” “smaller reporting company,” and “emerging growth company” in Rule 12b-2 of the Exchange Act.\nLarge accelerated filer\t\t☒\t\tAccelerated filer\t\t☐\nNon-accelerated filer\t\t☐\t\tSmaller reporting company\t\t☐\nEmerging growth company\t\t☐\n \nIf an emerging growth company, indicate by check mark if the Registrant has elected not to use the extended transition period for complying with any new or revised financial accounting standards provided pursuant to Section 13(a) of the Exchange Act. ☐\nIndicate by check mark whether the Registrant has filed a report on and attestation to its management’s assessment of the effectiveness of its internal control over financial reporting under Section 404(b) of the Sarbanes-Oxley Act (15 U.S.C. 7262(b)) by the registered public accounting firm that prepared or issued its audit report. ☒\nIf securities are registered pursuant to Section 12(b) of the Act, indicate by check mark whether the financial statements of the registrant included in the filing reflect the correction of an error to previously issued financial statements. ☐\nIndicate by check mark whether any of those error corrections are restatements that required a recovery analysis of incentive-based compensation received by any of the registrant’s executive officers during the relevant recovery period pursuant to §240.10D-1(b). ☐\nIndicate by check mark whether the Registrant is a shell company (as defined in Rule 12b-2 of the Act).\nYes  ☐     No  ☒\nThe aggregate market value of the voting and non-voting stock held by non-affiliates of the Registrant, as of March 29, 2024, the last business day of the Registrant’s most recently completed second fiscal quarter, was approximately $2,628,553,000,000. Solely for purposes of this disclosure, shares of common stock held by executive officers and directors of the Registrant as of such date have been excluded because such persons may be deemed to be affiliates. This determination of executive officers and directors as affiliates is not necessarily a conclusive determination for any other purposes.\n15,115,823,000 shares of common stock were issued and outstanding as of October 18, 2024.\nDOCUMENTS INCORPORATED BY REFERENCE\nPortions of the Registrant’s definitive proxy statement relating to its 2025 annual meeting of shareholders are incorporated by reference into Part III of this Annual Report on Form 10-K where indicated. The Registrant’s definitive proxy statement will be filed with the U.S. Securities and Exchange Commission within 120 days after the end of the fiscal year to which this report relates.\n \n\n\nApple Inc.\n\nForm 10-K\nFor the Fiscal Year Ended September 28, 2024\nTABLE OF CONTENTS\n\nPage\nPart I\nItem 1.\nBusiness\n1\nItem 1A.\nRisk Factors\n5\nItem 1B.\nUnresolved Staff Comments\n17\nItem 1C.\nCybersecurity\n17\nItem 2.\nProperties\n18\nItem 3.\nLegal Proceedings\n18\nItem 4.\nMine Safety Disclosures\n18\nPart II\nItem 5.\nMarket for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities\n19\nItem 6.\n[Reserved]\n20\nItem 7.\nManagement’s Discussion and Analysis of Financial Condition and Results of Operations\n21\nItem 7A.\nQuantitative and Qualitative Disclosures About Market Risk\n27\nItem 8.\nFinancial Statements and Supplementary Data\n28\nItem 9.\nChanges in and Disagreements with Accountants on Accounting and Financial Disclosure\n51\nItem 9A.\nControls and Procedures\n51\nItem 9B.\nOther Information\n52\nItem 9C.\nDisclosure Regarding Foreign Jurisdictions that Prevent Inspections\n52\nPart III\nItem 10.\nDirectors, Executive Officers and Corporate Governance\n52\nItem 11.\nExecutive Compensation\n52\nItem 12.\nSecurity Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters\n52\nItem 13.\nCertain Relationships and Related Transactions, and Director Independence\n52\nItem 14.\nPrincipal Accountant Fees and Services\n52\nPart IV\nItem 15.\nExhibit and Financial Statement Schedules\n53\nItem 16.\nForm 10-K Summary\n56\n \n\n\nThis Annual Report on Form 10-K (“Form 10-K”) contains forward-looking statements, within the meaning of the Private Securities Litigation Reform Act of 1995, that involve risks and uncertainties. Many of the forward-looking statements are located in Part I, Item 1 of this Form 10-K under the heading “Business” and Part II, Item 7 of this Form 10-K under the heading “Management’s Discussion and Analysis of Financial Condition and Results of Operations.” Forward-looking statements provide current expectations of future events based on certain assumptions and include any statement that does not directly relate to any historical or current fact. For example, statements in this Form 10-K regarding the potential future impact of macroeconomic conditions on the Company’s business and results of operations are forward-looking statements. Forward-looking statements can also be identified by words such as “future,” “anticipates,” “believes,” “estimates,” “expects,” “intends,” “plans,” “predicts,” “will,” “would,” “could,” “can,” “may,” and similar terms. Forward-looking statements are not guarantees of future performance and the Company’s actual results may differ significantly from the results discussed in the forward-looking statements. Factors that might cause such differences include, but are not limited to, those discussed in Part I, Item 1A of this Form 10-K under the heading “Risk Factors.” The Company assumes no obligation to revise or update any forward-looking statements for any reason, except as required by law.\nUnless otherwise stated, all information presented herein is based on the Company’s fiscal calendar, and references to particular years, quarters, months or periods refer to the Company’s fiscal years ended in September and the associated quarters, months and periods of those fiscal years. Each of the terms the “Company” and “Apple” as used herein refers collectively to Apple Inc. and its wholly owned subsidiaries, unless otherwise stated.\nPART I\nItem 1.    Business\nCompany Background\nThe Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company’s fiscal year is the 52- or 53-week period that ends on the last Saturday of September.\nProducts\niPhone\niPhone® is the Company’s line of smartphones based on its iOS operating system. The iPhone line includes iPhone 16 Pro, iPhone 16, iPhone 15, iPhone 14 and iPhone SE®.\nMac\nMac® is the Company’s line of personal computers based on its macOS® operating system. The Mac line includes laptops MacBook Air® and MacBook Pro®, as well as desktops iMac®, Mac mini®, Mac Studio® and Mac Pro®.\niPad\niPad® is the Company’s line of multipurpose tablets based on its iPadOS® operating system. The iPad line includes iPad Pro®, iPad Air®, iPad and iPad mini®.\nWearables, Home and Accessories\nWearables includes smartwatches, wireless headphones and spatial computers. The Company’s line of smartwatches, based on its watchOS® operating system, includes Apple Watch Ultra® 2, Apple Watch® Series 10 and Apple Watch SE®. The Company’s line of wireless headphones includes AirPods®, AirPods Pro®, AirPods Max® and Beats® products. Apple Vision Pro™ is the Company’s first spatial computer based on its visionOS™ operating system.\nHome includes Apple TV®, the Company’s media streaming and gaming device based on its tvOS® operating system, and HomePod® and HomePod mini®, high-fidelity wireless smart speakers.\nAccessories includes Apple-branded and third-party accessories.\nApple Inc. | 2024 Form 10-K | 1\n\nServices\nAdvertising\nThe Company’s advertising services include third-party licensing arrangements and the Company’s own advertising platforms.\nAppleCare\nThe Company offers a portfolio of fee-based service and support products under the AppleCare® brand. The offerings provide priority access to Apple technical support, access to the global Apple authorized service network for repair and replacement services, and in many cases additional coverage for instances of accidental damage or theft and loss, depending on the country and type of product.\nCloud Services\nThe Company’s cloud services store and keep customers’ content up-to-date and available across multiple Apple devices and Windows personal computers.\nDigital Content\nThe Company operates various platforms, including the App Store®, that allow customers to discover and download applications and digital content, such as books, music, video, games and podcasts.\nThe Company also offers digital content through subscription-based services, including Apple Arcade®, a game subscription service; Apple Fitness+SM, a personalized fitness service; Apple Music®, which offers users a curated listening experience with on-demand radio stations; Apple News+®, a subscription news and magazine service; and Apple TV+®, which offers exclusive original content and live sports.\nPayment Services\nThe Company offers payment services, including Apple Card®, a co-branded credit card, and Apple Pay®, a cashless payment service.\nSegments\nThe Company manages its business primarily on a geographic basis. The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific. Americas includes both North and South America. Europe includes European countries, as well as India, the Middle East and Africa. Greater China includes China mainland, Hong Kong and Taiwan. Rest of Asia Pacific includes Australia and those Asian countries not included in the Company’s other reportable segments. Although the reportable segments provide similar hardware and software products and similar services, each one is managed separately to better align with the location of the Company’s customers and distribution partners and the unique market dynamics of each geographic region.\nMarkets and Distribution\nThe Company’s customers are primarily in the consumer, small and mid-sized business, education, enterprise and government markets. The Company sells its products and resells third-party products in most of its major markets directly to customers through its retail and online stores and its direct sales force. The Company also employs a variety of indirect distribution channels, such as third-party cellular network carriers, wholesalers, retailers and resellers. During 2024, the Company’s net sales through its direct and indirect distribution channels accounted for 38% and 62%, respectively, of total net sales.\nCompetition\nThe markets for the Company’s products and services are highly competitive, and are characterized by aggressive price competition and resulting downward pressure on gross margins, frequent introduction of new products and services, short product life cycles, evolving industry standards, continual improvement in product price and performance characteristics, rapid adoption of technological advancements by competitors, and price sensitivity on the part of consumers and businesses. Many of the Company’s competitors seek to compete primarily through aggressive pricing and very low cost structures, and by imitating the Company’s products and infringing on its intellectual property.\nApple Inc. | 2024 Form 10-K | 2\n\nThe Company’s ability to compete successfully depends heavily on ensuring the continuing and timely introduction of innovative new products, services and technologies to the marketplace. The Company designs and develops nearly the entire solution for its products, including the hardware, operating system, numerous software applications and related services. Principal competitive factors important to the Company include price, product and service features (including security features), relative price and performance, product and service quality and reliability, design innovation, a strong third-party software and accessories ecosystem, marketing and distribution capability, service and support, and corporate reputation.\nThe Company is focused on expanding its market opportunities related to smartphones, personal computers, tablets, wearables and accessories, and services. The Company faces substantial competition in these markets from companies that have significant technical, marketing, distribution and other resources, as well as established hardware, software, and service offerings with large customer bases. In addition, some of the Company’s competitors have broader product lines, lower-priced products and a larger installed base of active devices. Competition has been particularly intense as competitors have aggressively cut prices and lowered product margins. Certain competitors have the resources, experience or cost structures to provide products at little or no profit or even at a loss. The Company’s services compete with business models that provide content to users for free and use illegitimate means to obtain third-party digital content and applications. The Company faces significant competition as competitors imitate the Company’s product features and applications within their products, or collaborate to offer integrated solutions that are more competitive than those they currently offer.\nSupply of Components\nAlthough most components essential to the Company’s business are generally available from multiple sources, certain components are currently obtained from single or limited sources. The Company also competes for various components with other participants in the markets for smartphones, personal computers, tablets, wearables and accessories. Therefore, many components used by the Company, including those that are available from multiple sources, are at times subject to industry-wide shortage and significant commodity pricing fluctuations.\nThe Company uses some custom components that are not commonly used by its competitors, and new products introduced by the Company often utilize custom components available from only one source. When a component or product uses new technologies, initial capacity constraints may exist until the suppliers’ yields have matured or their manufacturing capacities have increased. The continued availability of these components at acceptable prices, or at all, may be affected if suppliers decide to concentrate on the production of common components instead of components customized to meet the Company’s requirements.\nThe Company has entered into agreements for the supply of many components; however, there can be no guarantee that the Company will be able to extend or renew these agreements on similar terms, or at all.\nResearch and Development\nBecause the industries in which the Company competes are characterized by rapid technological advances, the Company’s ability to compete successfully depends heavily upon its ability to ensure a continual and timely flow of competitive products, services and technologies to the marketplace. The Company continues to develop new technologies to enhance existing products and services, and to expand the range of its offerings through research and development (“R&D”), licensing of intellectual property and acquisition of third-party businesses and technology.\nIntellectual Property\nThe Company currently holds a broad collection of intellectual property rights relating to certain aspects of its hardware, accessories, software and services. This includes patents, designs, copyrights, trademarks, trade secrets and other forms of intellectual property rights in the U.S. and various foreign countries. Although the Company believes the ownership of such intellectual property rights is an important factor in differentiating its business and that its success does depend in part on such ownership, the Company relies primarily on the innovative skills, technical competence and marketing abilities of its personnel.\nThe Company regularly files patent, design, copyright and trademark applications to protect innovations arising from its research, development, design and marketing, and is currently pursuing thousands of applications around the world. Over time, the Company has accumulated a large portfolio of issued and registered intellectual property rights around the world. No single intellectual property right is solely responsible for protecting the Company’s products and services. The Company believes the duration of its intellectual property rights is adequate relative to the expected lives of its products and services.\nIn addition to Company-owned intellectual property, many of the Company’s products and services are designed to include intellectual property owned by third parties. It may be necessary in the future to seek or renew licenses relating to various aspects of the Company’s products, processes and services. While the Company has generally been able to obtain such licenses on commercially reasonable terms in the past, there is no guarantee that such licenses could be obtained in the future on reasonable terms or at all.\nApple Inc. | 2024 Form 10-K | 3\n\nBusiness Seasonality and Product Introductions\nThe Company has historically experienced higher net sales in its first quarter compared to other quarters in its fiscal year due in part to seasonal holiday demand. Additionally, new product and service introductions can significantly impact net sales, cost of sales and operating expenses. The timing of product introductions can also impact the Company’s net sales to its indirect distribution channels as these channels are filled with new inventory following a product launch, and channel inventory of an older product often declines as the launch of a newer product approaches. Net sales can also be affected when consumers and distributors anticipate a product introduction.\nHuman Capital\nThe Company believes that its people play an important role in its success, and strives to attract, develop and retain the best talent. The Company works to create an inclusive, safe and supportive environment for all of its team members, so that its people can do the best work of their lives. As of September 28, 2024, the Company had approximately 164,000 full-time equivalent employees.\nCompensation and Benefits\nThe Company believes that compensation should be competitive and equitable, and should enable employees to share in the Company’s success. The Company recognizes its people are most likely to thrive when they have the resources to meet their needs and the time and support to succeed in their professional and personal lives. In support of this, the Company offers a wide variety of benefits for employees around the world, including health, wellness and time away.\nGrowth and Development\nThe Company invests in resources to help its people develop and achieve their career goals. The Company offers programs through Apple University on leadership, management and influence, as well as Apple culture and values. Team members can also take advantage of online classes for business, technical and personal development, as well as learning opportunities to support their well-being.\nWorkplace Practices and Policies\nThe Company is an equal opportunity employer committed to inclusion and diversity and to providing a workplace free of harassment or discrimination.\nInclusion and Diversity\nThe Company is committed to its vision to build and sustain a more inclusive workforce that is representative of the communities it serves. The Company continues to work to increase diverse representation at every level, foster an inclusive culture, and support equitable pay and access to opportunity for all employees.\nEngagement\nThe Company believes that open and honest communication among team members, managers and leaders helps create an open, collaborative work environment where everyone can contribute, grow and succeed. Team members are encouraged to come to their managers with questions, feedback or concerns, and the Company conducts surveys that gauge employee sentiment in areas like career development, manager performance and inclusivity.\nHealth and Safety\nThe Company is committed to protecting its team members everywhere it operates. The Company identifies potential workplace risks in order to develop measures to mitigate possible hazards. The Company supports employees with general safety, security and crisis management training, and by putting specific programs in place for those working in potentially high-hazard environments. Additionally, the Company works to protect the safety and security of its team members, visitors and customers through its global security team.\nApple Inc. | 2024 Form 10-K | 4\n\nAvailable Information\nThe Company’s Annual Reports on Form 10-K, Quarterly Reports on Form 10-Q, Current Reports on Form 8-K, and amendments to reports filed pursuant to Sections 13(a) and 15(d) of the Securities Exchange Act of 1934, as amended (the “Exchange Act”), are filed with the U.S. Securities and Exchange Commission (the “SEC”). Such reports and other information filed by the Company with the SEC are available free of charge at investor.apple.com/investor-relations/sec-filings/default.aspx when such reports are available on the SEC’s website. The Company periodically provides certain information for investors on its corporate website, www.apple.com, and its investor relations website, investor.apple.com. This includes press releases and other information about financial performance, information on environmental, social and governance matters, and details related to the Company’s annual meeting of shareholders. The information contained on the websites referenced in this Form 10-K is not incorporated by reference into this filing. Further, the Company’s references to website URLs are intended to be inactive textual references only.\nItem 1A.    Risk Factors\nThe Company’s business, reputation, results of operations, financial condition and stock price can be affected by a number of factors, whether currently known or unknown, including those described below. When any one or more of these risks materialize from time to time, the Company’s business, reputation, results of operations, financial condition and stock price can be materially and adversely affected.\nBecause of the following factors, as well as other factors affecting the Company’s results of operations and financial condition, past financial performance should not be considered to be a reliable indicator of future performance, and investors should not use historical trends to anticipate results or trends in future periods. This discussion of risk factors contains forward-looking statements.\nThis section should be read in conjunction with Part II, Item 7, “Management’s Discussion and Analysis of Financial Condition and Results of Operations” and the consolidated financial statements and accompanying notes in Part II, Item 8, “Financial Statements and Supplementary Data” of this Form 10-K.\nMacroeconomic and Industry Risks\nThe Company’s operations and performance depend significantly on global and regional economic conditions and adverse economic conditions can materially adversely affect the Company’s business, results of operations and financial condition.\nThe Company has international operations with sales outside the U.S. representing a majority of the Company’s total net sales. In addition, the Company’s global supply chain is large and complex and a majority of the Company’s supplier facilities, including manufacturing and assembly sites, are located outside the U.S. As a result, the Company’s operations and performance depend significantly on global and regional economic conditions.\nAdverse macroeconomic conditions, including slow growth or recession, high unemployment, inflation, tighter credit, higher interest rates, and currency fluctuations, can adversely impact consumer confidence and spending and materially adversely affect demand for the Company’s products and services. In addition, consumer confidence and spending can be materially adversely affected in response to changes in fiscal and monetary policy, financial market volatility, declines in income or asset values, and other economic factors.\nIn addition to an adverse impact on demand for the Company’s products and services, uncertainty about, or a decline in, global or regional economic conditions can have a significant impact on the Company’s suppliers, contract manufacturers, logistics providers, distributors, cellular network carriers and other channel partners, and developers. Potential outcomes include financial instability; inability to obtain credit to finance business operations; and insolvency.\nAdverse economic conditions can also lead to increased credit and collectibility risk on the Company’s trade receivables; the failure of derivative counterparties and other financial institutions; limitations on the Company’s ability to issue new debt; reduced liquidity; and declines in the fair values of the Company’s financial instruments. These and other impacts can materially adversely affect the Company’s business, results of operations, financial condition and stock price.\nApple Inc. | 2024 Form 10-K | 5\n\nThe Company’s business can be impacted by political events, trade and other international disputes, geopolitical tensions, conflict, terrorism, natural disasters, public health issues, industrial accidents and other business interruptions.\nPolitical events, trade and other international disputes, geopolitical tensions, conflict, terrorism, natural disasters, public health issues, industrial accidents and other business interruptions can have a material adverse effect on the Company and its customers, employees, suppliers, contract manufacturers, logistics providers, distributors, cellular network carriers and other channel partners.\nThe Company has a large, global business with sales outside the U.S. representing a majority of the Company’s total net sales, and the Company believes that it generally benefits from growth in international trade. Substantially all of the Company’s manufacturing is performed in whole or in part by outsourcing partners located primarily in China mainland, India, Japan, South Korea, Taiwan and Vietnam. Restrictions on international trade, such as tariffs and other controls on imports or exports of goods, technology or data, can materially adversely affect the Company’s business and supply chain. The impact can be particularly significant if these restrictive measures apply to countries and regions where the Company derives a significant portion of its revenues and/or has significant supply chain operations. Restrictive measures can increase the cost of the Company’s products and the components and raw materials that go into them, and can require the Company to take various actions, including changing suppliers, restructuring business relationships and operations, and ceasing to offer and distribute affected products, services and third-party applications to its customers. Changing the Company’s business and supply chain in accordance with new or changed restrictions on international trade can be expensive, time-consuming and disruptive to the Company’s operations. Such restrictions can be announced with little or no advance notice, which can create uncertainty, and the Company may not be able to effectively mitigate all adverse impacts from such measures. For example, tensions between governments, including the U.S. and China, have in the past led to tariffs and other restrictions affecting the Company’s business. If disputes and conflicts further escalate in the future, actions by governments in response could be significantly more severe and restrictive and could materially adversely affect the Company’s business.\nMany of the Company’s operations and facilities, as well as critical business operations of the Company’s suppliers and contract manufacturers, are in locations that are prone to earthquakes and other natural disasters. Global climate change is resulting in certain types of natural disasters and extreme weather occurring more frequently or with more intense effects. In addition, the Company’s and its suppliers’ operations and facilities are subject to the risk of interruption by fire, power shortages, nuclear power plant accidents and other industrial accidents, terrorist attacks and other hostile acts, ransomware and other cybersecurity attacks, labor disputes, public health issues and other events beyond the Company’s control. For example, global supply chains can be highly concentrated and geopolitical tensions or conflict could result in significant disruptions.\nSuch events can make it difficult or impossible for the Company to manufacture and deliver products to its customers, create delays and inefficiencies in the Company’s supply and manufacturing chain, result in slowdowns and outages to the Company’s service offerings, increase the Company’s costs, and negatively impact consumer spending and demand in affected areas.\nThe Company’s operations are also subject to the risks of industrial accidents at its suppliers and contract manufacturers. While the Company’s suppliers are required to maintain safe working environments and operations, an industrial accident could occur and could result in serious injuries or loss of life, disruption to the Company’s business, and harm to the Company’s reputation. Major public health issues, including pandemics such as the COVID-19 pandemic, have adversely affected, and could in the future materially adversely affect, the Company due to their impact on the global economy and demand for consumer products; the imposition of protective public safety measures, such as stringent employee travel restrictions and limitations on freight services and the movement of products between regions; and disruptions in the Company’s operations, supply chain and sales and distribution channels, resulting in interruptions to the supply of current products and offering of existing services, and delays in production ramps of new products and development of new services.\nFollowing any interruption to its business, the Company can require substantial recovery time, experience significant expenditures to resume operations, and lose significant sales. Because the Company relies on single or limited sources for the supply and manufacture of many critical components, a business interruption affecting such sources would exacerbate any negative consequences to the Company. While the Company maintains insurance coverage for certain types of losses, such insurance coverage may be insufficient to cover all losses that may arise.\nApple Inc. | 2024 Form 10-K | 6\n\nGlobal markets for the Company’s products and services are highly competitive and subject to rapid technological change, and the Company may be unable to compete effectively in these markets.\nThe Company’s products and services are offered in highly competitive global markets characterized by aggressive price competition and resulting downward pressure on gross margins, frequent introduction of new products and services, short product life cycles, evolving industry standards, continual improvement in product price and performance characteristics, rapid adoption of technological advancements by competitors, and price sensitivity on the part of consumers and businesses.\nThe Company’s ability to compete successfully depends heavily on ensuring the continuing and timely introduction of innovative new products, services and technologies to the marketplace. The Company designs and develops nearly the entire solution for its products, including the hardware, operating system, numerous software applications and related services. As a result, the Company must make significant investments in R&D. There can be no assurance these investments will achieve expected returns, and the Company may not be able to develop and market new products and services successfully.\nThe Company currently holds a significant number of patents, trademarks and copyrights and has registered, and applied to register, additional patents, trademarks and copyrights. In contrast, many of the Company’s competitors seek to compete primarily through aggressive pricing and very low cost structures, and by imitating the Company’s products and infringing on its intellectual property. Effective intellectual property protection is not consistently available in every country in which the Company operates. If the Company is unable to continue to develop and sell innovative new products with attractive margins or if competitors infringe on the Company’s intellectual property, the Company’s ability to maintain a competitive advantage could be materially adversely affected.\nThe Company has a minority market share in the global smartphone, personal computer and tablet markets. The Company faces substantial competition in these markets from companies that have significant technical, marketing, distribution and other resources, as well as established hardware, software and digital content supplier relationships. In addition, some of the Company’s competitors have broader product lines, lower-priced products and a larger installed base of active devices. Competition has been particularly intense as competitors have aggressively cut prices and lowered product margins. Certain competitors have the resources, experience or cost structures to provide products at little or no profit or even at a loss. Some of the markets in which the Company competes have from time to time experienced little to no growth or contracted overall.\nAdditionally, the Company faces significant competition as competitors imitate the Company’s product features and applications within their products or collaborate to offer solutions that are more competitive than those they currently offer. The Company also expects competition to intensify as competitors imitate the Company’s approach to providing components seamlessly within their offerings or work collaboratively to offer integrated solutions.\nThe Company’s services also face substantial competition, including from companies that have significant resources and experience and have established service offerings with large customer bases. The Company competes with business models that provide content to users for free. The Company also competes with illegitimate means to obtain third-party digital content and applications.\nThe Company’s business, results of operations and financial condition depend substantially on the Company’s ability to continually improve its products and services to maintain their functional and design advantages. There can be no assurance the Company will be able to continue to provide products and services that compete effectively.\nBusiness Risks\nTo remain competitive and stimulate customer demand, the Company must successfully manage frequent introductions and transitions of products and services.\nDue to the highly volatile and competitive nature of the markets and industries in which the Company competes, the Company must continually introduce new products, services and technologies, enhance existing products and services, effectively stimulate customer demand for new and upgraded products and services, and successfully manage the transition to these new and upgraded products and services. The success of new product and service introductions depends on a number of factors, including timely and successful development, market acceptance, the Company’s ability to manage the risks associated with new technologies and production ramp-up issues, the availability of application software or other third-party support for the Company’s products and services, the effective management of purchase commitments and inventory levels in line with anticipated product demand, the availability of products in appropriate quantities and at expected costs to meet anticipated demand, and the risk that new products and services may have quality or other defects or deficiencies. New products, services and technologies may replace or supersede existing offerings and may produce lower revenues and lower profit margins, which can materially adversely impact the Company’s business, results of operations and financial condition. There can be no assurance the Company will successfully manage future introductions and transitions of products and services.\nApple Inc. | 2024 Form 10-K | 7\n\nThe Company depends on component and product manufacturing and logistical services provided by outsourcing partners, many of which are located outside of the U.S.\nSubstantially all of the Company’s manufacturing is performed in whole or in part by outsourcing partners located primarily in China mainland, India, Japan, South Korea, Taiwan and Vietnam, and a significant concentration of this manufacturing is currently performed by a small number of outsourcing partners, often in single locations. The Company has also outsourced much of its transportation and logistics management. While these arrangements can lower operating costs, they also reduce the Company’s direct control over production and distribution. Such diminished control has from time to time and may in the future have an adverse effect on the quality or quantity of products manufactured or services provided, or adversely affect the Company’s flexibility to respond to changing conditions. Although arrangements with these partners may contain provisions for product defect expense reimbursement, the Company generally remains responsible to the consumer for warranty and out-of-warranty service in the event of product defects and experiences unanticipated product defect liabilities from time to time. While the Company relies on its partners to adhere to its supplier code of conduct, violations of the supplier code of conduct occur from time to time and can materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nThe Company relies on single-source outsourcing partners in the U.S., Asia and Europe to supply and manufacture many components, and on outsourcing partners primarily located in Asia, for final assembly of substantially all of the Company’s hardware products. Any failure of these partners to perform can have a negative impact on the Company’s cost or supply of components or finished goods. In addition, manufacturing or logistics in these locations or transit to final destinations can be disrupted for a variety of reasons, including natural and man-made disasters, information technology system failures, commercial disputes, economic, business, labor, environmental, public health or political issues, trade and other international disputes, geopolitical tensions, or conflict.\nThe Company has invested in manufacturing process equipment, much of which is held at certain of its outsourcing partners, and has made prepayments to certain of its suppliers associated with long-term supply agreements. While these arrangements help ensure the supply of components and finished goods, if these outsourcing partners or suppliers experience severe financial problems or other disruptions in their business, such continued supply can be disrupted or terminated, and the recoverability of manufacturing process equipment or prepayments can be negatively impacted.\nChanges or additions to the Company’s supply chain require considerable time and resources and involve significant risks and uncertainties, including exposure to additional regulatory and operational risks.\nFuture operating results depend upon the Company’s ability to obtain components in sufficient quantities on commercially reasonable terms.\nBecause the Company currently obtains certain components from single or limited sources, the Company is subject to significant supply and pricing risks. Many components, including those that are available from multiple sources, are at times subject to industry-wide shortages and significant commodity pricing fluctuations that can materially adversely affect the Company’s business, results of operations and financial condition. For example, the global semiconductor industry has in the past experienced high demand and shortages of supply, which adversely affected the Company’s ability to obtain sufficient quantities of components and products on commercially reasonable terms, or at all. Such disruptions could occur in the future. While the Company has entered into agreements for the supply of many components, there can be no assurance the Company will be able to extend or renew these agreements on similar terms, or at all. In addition, component suppliers may suffer from poor financial conditions, which can lead to business failure for the supplier or consolidation within a particular industry, further limiting the Company’s ability to obtain sufficient quantities of components on commercially reasonable terms, or at all. Therefore, the Company remains subject to significant risks of supply shortages and price increases that can materially adversely affect its business, results of operations and financial condition.\nThe Company’s new products often utilize custom components available from only one source. When a component or product uses new technologies, initial capacity constraints may exist until the suppliers’ yields have matured or their manufacturing capacities have increased. The continued availability of these components at acceptable prices, or at all, can be affected for any number of reasons, including if suppliers decide to concentrate on the production of common components instead of components customized to meet the Company’s requirements. When the Company’s supply of components for a new or existing product has been delayed or constrained, or when an outsourcing partner has delayed shipments of completed products to the Company, the Company’s business, results of operations and financial condition have been adversely affected and future delays or constraints could materially adversely affect the Company’s business, results of operations and financial condition. The Company’s business and financial performance could also be materially adversely affected depending on the time required to obtain sufficient quantities from the source, or to identify and obtain sufficient quantities from an alternative source.\nApple Inc. | 2024 Form 10-K | 8\n\nThe Company’s products and services may be affected from time to time by design and manufacturing defects that could materially adversely affect the Company’s business and result in harm to the Company’s reputation.\nThe Company offers complex hardware and software products and services that can be affected by design and manufacturing defects. Sophisticated operating system software and applications, such as those offered by the Company, often have issues that can unexpectedly interfere with the intended operation of hardware or software products and services. Defects can also exist in components and products the Company purchases from third parties. Component defects could make the Company’s products unsafe and create a risk of environmental or property damage and personal injury. These risks may increase as the Company’s products are introduced into specialized applications, including health. In addition, the Company’s service offerings can have quality issues and from time to time experience outages, service slowdowns or errors. As a result, from time to time the Company’s services have not performed as anticipated and may not meet customer expectations. The introduction of new and complex technologies, such as artificial intelligence features, can increase these and other safety risks, including exposing users to harmful, inaccurate or other negative content and experiences. There can be no assurance the Company will be able to detect and fix all issues and defects in the hardware, software and services it offers. Failure to do so can result in widespread technical and performance issues affecting the Company’s products and services. Errors, bugs and vulnerabilities can be exploited by third parties, compromising the safety and security of a user’s device. In addition, the Company can be exposed to product liability claims, recalls, product replacements or modifications, write-offs of inventory, property, plant and equipment or intangible assets, and significant warranty and other expenses, including litigation costs and regulatory fines. Quality problems can adversely affect the experience for users of the Company’s products and services, and result in harm to the Company’s reputation, loss of competitive advantage, poor market acceptance, reduced demand for products and services, delay in new product and service introductions and lost sales.\nThe Company is exposed to the risk of write-downs on the value of its inventory and other assets, in addition to purchase commitment cancellation risk.\nThe Company records a write-down for product and component inventories that have become obsolete or exceed anticipated demand, or for which cost exceeds net realizable value. The Company also accrues necessary cancellation fee reserves for orders of excess products and components. The Company reviews long-lived assets, including capital assets held at its suppliers’ facilities and inventory prepayments, for impairment whenever events or circumstances indicate the assets may not be recoverable. If the Company determines that an impairment has occurred, it records a write-down equal to the amount by which the carrying value of the asset exceeds its fair value. Although the Company believes its inventory, capital assets, inventory prepayments and other assets and purchase commitments are currently recoverable, there can be no assurance the Company will not incur write-downs, fees, impairments and other charges given the rapid and unpredictable pace of product obsolescence in the industries in which the Company competes.\nThe Company orders components for its products and builds inventory in advance of product announcements and shipments. Manufacturing purchase obligations cover the Company’s forecasted component and manufacturing requirements, typically for periods up to 150 days. Because the Company’s markets are volatile, competitive and subject to rapid technology and price changes, there is a risk the Company will forecast incorrectly and order or produce excess or insufficient amounts of components or products, or not fully utilize firm purchase commitments.\nThe Company relies on access to third-party intellectual property, which may not be available to the Company on commercially reasonable terms, or at all.\nThe Company’s products and services are designed to include intellectual property owned by third parties, which requires licenses from those third parties. In addition, because of technological changes in the industries in which the Company currently competes or in the future may compete, current extensive patent coverage and the rapid rate of issuance of new patents, the Company’s products and services can unknowingly infringe existing patents or intellectual property rights of others. From time to time, the Company has been notified that it may be infringing certain patents or other intellectual property rights of third parties. Based on experience and industry practice, the Company believes licenses to such third-party intellectual property can generally be obtained on commercially reasonable terms. However, there can be no assurance the necessary licenses can be obtained on commercially reasonable terms or at all. Failure to obtain the right to use third-party intellectual property, or to use such intellectual property on commercially reasonable terms, can require the Company to modify certain products, services or features or preclude the Company from selling certain products or services, or otherwise have a material adverse impact on the Company’s business, results of operations and financial condition.\nApple Inc. | 2024 Form 10-K | 9\n\nThe Company’s future performance depends in part on support from third-party software developers.\nThe Company believes decisions by customers to purchase its hardware products depend in part on the availability of third-party software applications and services. There can be no assurance third-party developers will continue to develop and maintain software applications and services for the Company’s products. If third-party software applications and services cease to be developed and maintained for the Company’s products, customers may choose not to buy the Company’s products.\nThe Company believes the availability of third-party software applications and services for its products depends in part on the developers’ perception and analysis of the relative benefits of developing, maintaining and upgrading such software and services for the Company’s products compared to competitors’ platforms, such as Android for smartphones and tablets, Windows for personal computers and tablets, and PlayStation, Nintendo and Xbox for gaming platforms. This analysis may be based on factors such as the market position of the Company and its products, the anticipated revenue that may be generated, expected future growth of product sales, and the costs of developing such applications and services.\nThe Company’s minority market share in the global smartphone, personal computer and tablet markets can make developers less inclined to develop or upgrade software for the Company’s products and more inclined to devote their resources to developing and upgrading software for competitors’ products with larger market share. When developers focus their efforts on these competing platforms, the availability and quality of applications for the Company’s devices can suffer.\nThe Company relies on the continued availability and development of compelling and innovative software applications for its products. The Company’s products and operating systems are subject to rapid technological change, and when third-party developers are unable to or choose not to keep up with this pace of change, their applications can fail to take advantage of these changes to deliver improved customer experiences, can operate incorrectly, and can result in dissatisfied customers and lower customer demand for the Company’s products.\nThe Company distributes third-party applications for its products through the App Store. For the vast majority of applications, developers keep all of the revenue they generate on the App Store. Where applicable, the Company retains a commission from sales of applications and sales of digital services or goods initiated within an application. From time to time, the Company has made changes to its products and services, including taking actions in response to litigation, competition, market conditions and legal and regulatory requirements, and expects to make further business changes in the future. For example, in the U.S., the Company has implemented changes to how developers communicate with consumers within apps on the U.S. storefront of the iOS and iPadOS App Store regarding alternative purchasing mechanisms. The Company has also implemented changes to iOS, iPadOS, the App Store and Safari® in the European Union (“EU”) as it seeks to comply with the Digital Markets Act (the “DMA”), including new business terms and alternative fee structures for iOS and iPadOS apps, alternative methods of distribution for iOS and iPadOS apps, alternative payment processing for apps across the Company’s operating systems, and additional tools and application programming interfaces (“APIs”) for developers. Changes to the Company’s products and services could materially adversely affect the Company’s business, results of operations and financial condition, including if such business changes result in reduced App Store or other sales, reductions in the rate of the commission that the Company retains on such sales, or if the rate of the commission is otherwise narrowed in scope or eliminated.\nFailure to obtain or create digital content that appeals to the Company’s customers, or to make such content available on commercially reasonable terms, could have a material adverse impact on the Company’s business, results of operations and financial condition.\nThe Company contracts with numerous third parties to offer their digital content to customers. This includes the right to sell, or offer subscriptions to, third-party content, as well as the right to incorporate specific content into the Company’s own services. The licensing or other distribution arrangements for this content can be for relatively short time periods and do not guarantee the continuation or renewal of these arrangements on commercially reasonable terms, or at all. Some third-party content providers and distributors currently or in the future may offer competing products and services, and can take actions to make it difficult or impossible for the Company to license or otherwise distribute their content. Other content owners, providers or distributors may seek to limit the Company’s access to, or increase the cost of, such content. The Company may be unable to continue to offer a wide variety of content at commercially reasonable prices with acceptable usage rules.\nThe Company also produces its own digital content, which can be costly to produce due to intense and increasing competition for talent, content and subscribers, and may fail to appeal to the Company’s customers.\nSome third-party digital content providers require the Company to provide digital rights management and other security solutions. If requirements change, the Company may have to develop or license new technology to provide these solutions. There can be no assurance the Company will be able to develop or license such solutions at a reasonable cost and in a timely manner.\nApple Inc. | 2024 Form 10-K | 10\n\nThe Company’s success depends largely on the talents and efforts of its team members, the continued service and availability of highly skilled employees, including key personnel, and the Company’s ability to nurture its distinctive and inclusive culture.\nMuch of the Company’s future success depends on the talents and efforts of its team members and the continued availability and service of key personnel, including its Chief Executive Officer, executive team and other highly skilled employees. Experienced personnel in the technology industry are in high demand and competition for their talents is intense, especially in Silicon Valley, where most of the Company’s key personnel are located. In addition to intense competition for talent, workforce dynamics are constantly evolving. If the Company does not manage changing workforce dynamics effectively, it could materially adversely affect the Company’s culture, reputation and operational flexibility.\nThe Company believes that its distinctive and inclusive culture is a significant driver of its success. If the Company is unable to nurture its culture, it could materially adversely affect the Company’s ability to recruit and retain the highly skilled employees who are critical to its success, and could otherwise materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nThe Company depends on the performance of carriers, wholesalers, retailers and other resellers.\nThe Company distributes its products and certain of its services through cellular network carriers, wholesalers, retailers and resellers, many of which distribute products and services from competitors. The Company also sells its products and services and resells third-party products in most of its major markets directly to consumers, small and mid-sized businesses, and education, enterprise and government customers through its retail and online stores and its direct sales force.\nSome carriers providing cellular network service for the Company’s products offer financing, installment payment plans or subsidies for users’ purchases of the device. There can be no assurance such offers will be continued at all or in the same amounts.\nThe Company has invested and will continue to invest in programs to enhance reseller sales, including staffing selected resellers’ stores with Company employees and contractors, and improving product placement displays. These programs can require a substantial investment while not assuring return or incremental sales. The financial condition of these resellers could weaken, these resellers could stop distributing the Company’s products, or uncertainty regarding demand for some or all of the Company’s products could cause resellers to reduce their ordering and marketing of the Company’s products.\nThe Company’s business and reputation are impacted by information technology system failures and network disruptions.\nThe Company and its global supply chain are dependent on complex information technology systems and are exposed to information technology system failures or network disruptions caused by natural disasters, accidents, power disruptions, telecommunications failures, acts of terrorism or war, computer viruses, physical or electronic break-ins, ransomware or other cybersecurity incidents, or other events or disruptions. System upgrades, redundancy and other continuity measures may be ineffective or inadequate, and the Company’s or its vendors’ business continuity and disaster recovery planning may not be sufficient for all eventualities. Such failures or disruptions can adversely impact the Company’s business by, among other things, preventing access to the Company’s online services, interfering with customer transactions or impeding the manufacturing and shipping of the Company’s products. These events could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nLosses or unauthorized access to or releases of confidential information, including personal information, could subject the Company to significant reputational, financial, legal and operational consequences.\nThe Company’s business requires it to use and store confidential information, including personal information with respect to the Company’s customers and employees. The Company devotes significant resources to systems and data security, including through the use of encryption and other security measures intended to protect its systems and data. But these measures cannot provide absolute security, and losses or unauthorized access to or releases of confidential information occur and could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nThe Company’s business also requires it to share confidential information with suppliers and other third parties. The Company relies on global suppliers that are also exposed to ransomware and other malicious attacks that can disrupt business operations. Although the Company takes steps to secure confidential information that is provided to or accessible by third parties working on the Company’s behalf, such measures are not always effective and losses or unauthorized access to, or releases of, confidential information occur. Such incidents and other malicious attacks could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nApple Inc. | 2024 Form 10-K | 11\n\nThe Company experiences malicious attacks and other attempts to gain unauthorized access to its systems on a regular basis. These attacks seek to compromise the confidentiality, integrity or availability of confidential information or disrupt normal business operations, and can, among other things, impair the Company’s ability to attract and retain customers for its products and services, impact the Company’s stock price, materially damage commercial relationships, and expose the Company to litigation or government investigations, which can result in penalties, fines or judgments against the Company. Globally, attacks are expected to continue accelerating in both frequency and sophistication with increasing use by actors of tools and techniques that are designed to circumvent controls, avoid detection, and remove or obfuscate forensic evidence, all of which hinders the Company’s ability to identify, investigate and recover from incidents. In addition, attacks against the Company and its customers can escalate during periods of geopolitical tensions or conflict.\nAlthough malicious attacks perpetrated to gain access to confidential information, including personal information, affect many companies across various industries, the Company is at a relatively greater risk of being targeted because of its high profile and the value of the confidential information it creates, owns, manages, stores and processes.\nThe Company has implemented systems and processes intended to secure its information technology systems and prevent unauthorized access to or loss of sensitive data, and mitigate the impact of unauthorized access, including through the use of encryption and authentication technologies. As with all companies, these security measures may not be sufficient for all eventualities and are vulnerable to hacking, ransomware attacks, employee error, malfeasance, system error, faulty password management or other irregularities. For example, third parties can fraudulently induce the Company’s or its suppliers’ and other third parties’ employees or customers into disclosing usernames, passwords or other sensitive information, which can, in turn, be used for unauthorized access to the Company’s or such suppliers’ or third parties’ systems and services. To help protect customers and the Company, the Company deploys and makes available technologies like multifactor authentication, monitors its services and systems for unusual activity and may freeze accounts under suspicious circumstances, which, among other things, can result in the delay or loss of customer orders or impede customer access to the Company’s products and services.\nWhile the Company maintains insurance coverage that is intended to address certain aspects of data security risks, such insurance coverage may be insufficient to cover all losses or all types of claims that may arise.\nInvestment in new business strategies and acquisitions could disrupt the Company’s ongoing business, present risks not originally contemplated and materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nThe Company has invested, and in the future may invest, in new business strategies or acquisitions. Such endeavors may involve significant risks and uncertainties, including distraction of management from current operations, greater-than-expected liabilities and expenses, economic, political, legal and regulatory challenges associated with operating in new businesses, regions or countries, inadequate return on capital, potential impairment of tangible and intangible assets, and significant write-offs. Investment and acquisition transactions are exposed to additional risks, including failing to obtain required regulatory approvals on a timely basis or at all, or the imposition of onerous conditions that could delay or prevent the Company from completing a transaction or otherwise limit the Company’s ability to fully realize the anticipated benefits of a transaction. These new ventures are inherently risky and may not be successful. The failure of any significant investment could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nThe Company’s retail stores are subject to numerous risks and uncertainties.\nThe Company’s retail operations are subject to many factors that pose risks and uncertainties and could adversely impact the Company’s business, results of operations and financial condition, including macroeconomic factors that could have an adverse effect on general retail activity. Other factors include the Company’s ability to: manage costs associated with retail store construction and operation; manage relationships with existing retail partners; manage costs associated with fluctuations in the value of retail inventory; and obtain and renew leases in quality retail locations at a reasonable cost.\nApple Inc. | 2024 Form 10-K | 12\n\nLegal and Regulatory Compliance Risks\nThe Company’s business, results of operations and financial condition could be adversely impacted by unfavorable results of legal proceedings or government investigations.\nThe Company is subject to various claims, legal proceedings and government investigations that have arisen in the ordinary course of business and have not yet been fully resolved, and new matters may arise in the future. In addition, agreements entered into by the Company sometimes include indemnification provisions which can subject the Company to costs and damages in the event of a claim against an indemnified third party. The number of claims, legal proceedings and government investigations involving the Company, and the alleged magnitude of such claims, proceedings and government investigations, has generally increased over time and may continue to increase.\nThe Company has faced and continues to face a significant number of patent claims relating to its cellular-enabled products, and new claims may arise in the future, including as a result of new legal or regulatory frameworks. For example, technology and other patent-holding companies frequently assert their patents and seek royalties and often enter into litigation based on allegations of patent infringement or other violations of intellectual property rights. The Company is vigorously defending infringement actions in courts in several U.S. jurisdictions, as well as internationally in various countries. The plaintiffs in these actions frequently seek broad injunctive relief and substantial damages.\nRegardless of the merit of particular claims, defending against litigation or responding to government investigations can be expensive, time-consuming and disruptive to the Company’s operations. In recognition of these considerations, the Company may enter into agreements or other arrangements to settle litigation and resolve such challenges. There can be no assurance such agreements can be obtained on acceptable terms or that litigation will not occur. These agreements can also significantly increase the Company’s cost of sales and operating expenses and require the Company to change its business practices and limit the Company’s ability to offer certain products and services.\nThe outcome of litigation or government investigations is inherently uncertain. If one or more legal matters were resolved against the Company or an indemnified third party in a reporting period for amounts above management’s expectations, the Company’s results of operations and financial condition for that reporting period could be materially adversely affected. Further, such an outcome can result in significant monetary damages, disgorgement of revenue or profits, remedial corporate measures or injunctive relief against the Company, and has from time to time required, and can in the future require, the Company to change its business practices and limit the Company’s ability to develop, manufacture, use, import or offer for sale certain products and services, all of which could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nWhile the Company maintains insurance coverage for certain types of claims, such insurance coverage may be insufficient to cover all losses or all types of claims that may arise.\nThe Company is subject to complex and changing laws and regulations worldwide, which exposes the Company to potential liabilities, increased costs and other adverse effects on the Company’s business.\nThe Company’s global operations are subject to complex and changing laws and regulations on subjects, including antitrust; privacy, data security and data localization; consumer protection; advertising, sales, billing and e-commerce; financial services and technology; product liability; intellectual property ownership and infringement; digital platforms; machine learning and artificial intelligence; internet, telecommunications and mobile communications; media, television, film and digital content; availability of third-party software applications and services; labor and employment; anticorruption; import, export and trade; foreign exchange controls and cash repatriation restrictions; anti–money laundering; foreign ownership and investment; tax; and environmental, health and safety, including electronic waste, recycling, product design and climate change.\nCompliance with these laws and regulations is onerous and expensive. New and changing laws and regulations can adversely affect the Company’s business by increasing the Company’s costs, limiting the Company’s ability to offer a product, service or feature to customers, imposing changes to the design of the Company’s products and services, impacting customer demand for the Company’s products and services, and requiring changes to the Company’s business or supply chain. New and changing laws and regulations can also create uncertainty about how such laws and regulations will be interpreted and applied. These risks and costs may increase as the Company’s products and services are introduced into specialized applications, including health and financial services, or as the Company expands the use of technologies, such as machine learning and artificial intelligence features, and must navigate new legal, regulatory and ethical considerations relating to such technologies. The Company has implemented policies and procedures designed to ensure compliance with applicable laws and regulations, but there can be no assurance the Company’s employees, contractors or agents will not violate such laws and regulations or the Company’s policies and procedures. If the Company is found to have violated laws and regulations, it could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nApple Inc. | 2024 Form 10-K | 13\n\nRegulatory changes and other actions that materially adversely affect the Company’s business may be announced with little or no advance notice and the Company may not be able to effectively mitigate all adverse impacts from such measures. For example, the Company is subject to changing regulations relating to the export and import of its products. Although the Company has programs, policies and procedures in place that are designed to satisfy regulatory requirements, there can be no assurance that such policies and procedures will be effective in preventing a violation or a claim of a violation. As a result, the Company’s products could be banned, delayed or prohibited from importation, which could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nExpectations relating to environmental, social and governance considerations and related reporting obligations expose the Company to potential liabilities, increased costs, reputational harm, and other adverse effects on the Company’s business.\nMany governments, regulators, investors, employees, customers and other stakeholders are increasingly focused on environmental, social and governance considerations relating to businesses, including climate change and greenhouse gas emissions, human and civil rights, and diversity, equity and inclusion. In addition, the Company makes statements about its goals and initiatives through its various non-financial reports, information provided on its website, press statements and other communications. Responding to these environmental, social and governance considerations and implementation of the Company’s announced goals and initiatives involves risks and uncertainties, requires investments, and depends in part on third-party performance or data that is outside the Company’s control. The Company cannot guarantee that it will achieve its announced environmental, social and governance goals and initiatives. In addition, some stakeholders may disagree with the Company’s goals and initiatives. Any failure, or perceived failure, by the Company to achieve its goals, further its initiatives, adhere to its public statements, comply with federal, state and international environmental, social and governance laws and regulations, or meet evolving and varied stakeholder expectations and standards could result in legal and regulatory proceedings against the Company and materially adversely affect the Company’s business, reputation, results of operations, financial condition and stock price.\nThe technology industry, including, in some instances, the Company, is subject to intense media, political and regulatory scrutiny, which exposes the Company to increasing regulation, government investigations, legal actions and penalties.\nFrom time to time, the Company has made changes to its App Store, including actions taken in response to litigation, competition, market conditions and legal and regulatory requirements. The Company expects to make further business changes in the future. For example, in the U.S. the Company has implemented changes to how developers communicate with consumers within apps on the U.S. storefront of the iOS and iPadOS App Store regarding alternative purchasing mechanisms.\nThe Company has also implemented changes to iOS, iPadOS, the App Store and Safari in the EU as it seeks to comply with the DMA, including new business terms and alternative fee structures for iOS and iPadOS apps, alternative methods of distribution for iOS and iPadOS apps, alternative payment processing for apps across the Company’s operating systems, and additional tools and APIs for developers. The Company has also continued to make changes to its compliance plan in response to feedback and engagement with the European Commission (the “Commission”). Although the Company’s compliance plan is intended to address the DMA’s obligations, it has been challenged by the Commission and may be challenged further by private litigants. The DMA provides for significant fines and penalties for noncompliance, and other jurisdictions may seek to require the Company to make changes to its business. While the changes introduced by the Company in the EU are intended to reduce new privacy and security risks that the DMA poses to EU users, many risks will remain.\nThe Company is also currently subject to antitrust investigations and litigation in various jurisdictions around the world, which can result in legal proceedings and claims against the Company that could, individually or in the aggregate, have a materially adverse impact on the Company’s business, results of operations and financial condition. For example, the Company is subject to civil antitrust lawsuits in the U.S. alleging monopolization or attempted monopolization in the markets for “performance smartphones” and “smartphones” generally in violation of U.S. antitrust laws. In addition, the Company is the subject of investigations in Europe and other jurisdictions relating to App Store terms and conditions. If such investigations or litigation are resolved against the Company, the Company can be exposed to significant fines and may be required to make further changes to its business practices, all of which could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nFurther, the Company has commercial relationships with other companies in the technology industry that are or may become subject to investigations and litigation that, if resolved against those other companies, could materially adversely affect the Company’s commercial relationships with those business partners and materially adversely affect the Company’s business, results of operations and financial condition. For example, the Company earns revenue from licensing arrangements with Google LLC and other companies to offer their search services on the Company’s platforms and applications, and certain of these arrangements are currently subject to government investigations and legal proceedings.\nApple Inc. | 2024 Form 10-K | 14\n\nThere can be no assurance the Company’s business will not be materially adversely affected, individually or in the aggregate, by the outcomes of such investigations, litigation or changes to laws and regulations in the future. Changes to the Company’s business practices to comply with new laws and regulations or in connection with other legal proceedings can negatively impact the reputation of the Company’s products for privacy and security and otherwise adversely affect the experience for users of the Company’s products and services, and result in harm to the Company’s reputation, loss of competitive advantage, poor market acceptance, reduced demand for products and services, and lost sales.\nThe Company’s business is subject to a variety of U.S. and international laws, rules, policies and other obligations regarding data protection.\nThe Company is subject to an increasing number of federal, state and international laws relating to the collection, use, retention, security and transfer of various types of personal information. In many cases, these laws apply not only to third-party transactions, but also restrict transfers of personal information among the Company and its international subsidiaries. Several jurisdictions have passed laws in this area, and additional jurisdictions are considering imposing additional restrictions or have laws that are pending. These laws continue to develop and may be inconsistent from jurisdiction to jurisdiction. Complying with emerging and changing requirements causes the Company to incur substantial costs and has required and may in the future require the Company to change its business practices. Noncompliance could result in significant penalties or legal liability.\nThe Company makes statements about its use and disclosure of personal information through its privacy policy, information provided on its website, press statements and other privacy notices provided to customers. Any failure by the Company to comply with these public statements or with federal, state or international privacy or data protection laws and regulations could result in inquiries or proceedings against the Company by governmental entities or others. In addition to reputational impacts, penalties could include ongoing audit requirements and significant legal liability.\nIn addition to the risks generally relating to the collection, use, retention, security and transfer of personal information, the Company is also subject to specific obligations relating to information considered sensitive under applicable laws, such as health data, financial data and biometric data. Health data and financial data are subject to additional privacy, security and breach notification requirements, and the Company is subject to audit by governmental authorities regarding the Company’s compliance with these obligations. If the Company fails to adequately comply with these rules and requirements, or if health data or financial data is handled in a manner not permitted by law or under the Company’s agreements with healthcare or financial institutions, the Company can be subject to litigation or government investigations, and can be liable for associated investigatory expenses, and can also incur significant fees or fines.\nPayment card data is also subject to additional requirements. Under payment card rules and obligations, if cardholder information is potentially compromised, the Company can be liable for associated investigatory expenses and can also incur significant fees or fines if the Company fails to follow payment card industry data security standards. The Company could also experience a significant increase in payment card transaction costs or lose the ability to process payment cards if it fails to follow payment card industry data security standards, which could materially adversely affect the Company’s business, reputation, results of operations and financial condition.\nFinancial Risks\nThe Company expects its quarterly net sales and results of operations to fluctuate.\nThe Company’s profit margins vary across its products, services, geographic segments and distribution channels. For example, the gross margins on the Company’s products and services vary significantly and can change over time. The Company’s gross margins are subject to volatility and downward pressure due to a variety of factors, including: continued industry-wide global product pricing pressures and product pricing actions that the Company may take in response to such pressures; increased competition; the Company’s ability to effectively stimulate demand for certain of its products and services; compressed product life cycles; supply shortages; potential increases in the cost of components, outside manufacturing services, and developing, acquiring and delivering content for the Company’s services; the Company’s ability to manage product quality and warranty costs effectively; shifts in the mix of products and services, or in the geographic, currency or channel mix, including to the extent that regulatory changes require the Company to modify its product and service offerings; fluctuations in foreign exchange rates; inflation and other macroeconomic pressures; and the introduction of new products or services, including new products or services with lower profit margins. These and other factors could have a materially adverse impact on the Company’s results of operations and financial condition.\nThe Company has historically experienced higher net sales in its first quarter compared to other quarters in its fiscal year due in part to seasonal holiday demand. Additionally, new product and service introductions can significantly impact net sales, cost of sales and operating expenses. Further, the Company generates a significant portion of its net sales from a single product and a decline in demand for that product could significantly impact quarterly net sales. The Company could also be subject to unexpected developments, such as lower-than-anticipated demand for the Company’s products or services, issues with new product or service introductions, information technology system failures or network disruptions, or failure of one of the Company’s logistics, supply or manufacturing partners.\nApple Inc. | 2024 Form 10-K | 15\n\nThe Company’s financial performance is subject to risks associated with changes in the value of the U.S. dollar relative to local currencies.\nThe Company’s primary exposure to movements in foreign exchange rates relates to non–U.S. dollar–denominated sales, cost of sales and operating expenses worldwide. Gross margins on the Company’s products in foreign countries and on products that include components obtained from foreign suppliers have in the past been adversely affected and could in the future be materially adversely affected by foreign exchange rate fluctuations.\nThe weakening of foreign currencies relative to the U.S. dollar adversely affects the U.S. dollar value of the Company’s foreign currency–denominated sales and earnings, and generally leads the Company to raise international pricing, potentially reducing demand for the Company’s products. In some circumstances, for competitive or other reasons, the Company may decide not to raise international pricing to offset the U.S. dollar’s strengthening, which would adversely affect the U.S. dollar value of the gross margins the Company earns on foreign currency–denominated sales.\nConversely, a strengthening of foreign currencies relative to the U.S. dollar, while generally beneficial to the Company’s foreign currency–denominated sales and earnings, could cause the Company to reduce international pricing or incur losses on its foreign currency derivative instruments, thereby limiting the benefit. Additionally, strengthening of foreign currencies may increase the Company’s cost of product components denominated in those currencies, thus adversely affecting gross margins.\nThe Company uses derivative instruments, such as foreign currency forward and option contracts, to hedge certain exposures to fluctuations in foreign exchange rates. The use of such hedging activities may not be effective to offset any, or more than a portion, of the adverse financial effects of unfavorable movements in foreign exchange rates over the limited time the hedges are in place.\nThe Company is exposed to credit risk and fluctuations in the values of its investment portfolio.\nThe Company’s investments can be negatively affected by changes in liquidity, credit deterioration, financial results, market and economic conditions, political risk, sovereign risk, interest rate fluctuations or other factors. As a result, the value and liquidity of the Company’s cash, cash equivalents and marketable securities may fluctuate substantially. Although the Company has not realized significant losses on its cash, cash equivalents and marketable securities, future fluctuations in their value could result in significant losses and could have a material adverse impact on the Company’s results of operations and financial condition.\nThe Company is exposed to credit risk on its trade accounts receivable, vendor non-trade receivables and prepayments related to long-term supply agreements, and this risk is heightened during periods when economic conditions worsen.\nThe Company distributes its products and certain of its services through third-party cellular network carriers, wholesalers, retailers and resellers. The Company also sells its products and services directly to small and mid-sized businesses and education, enterprise and government customers. A substantial majority of the Company’s outstanding trade receivables are not covered by collateral, third-party bank support or financing arrangements, or credit insurance, and a significant portion of the Company’s trade receivables can be concentrated within cellular network carriers or other resellers. The Company’s exposure to credit and collectibility risk on its trade receivables is higher in certain international markets and its ability to mitigate such risks may be limited. The Company also has unsecured vendor non-trade receivables resulting from purchases of components by outsourcing partners and other vendors that manufacture subassemblies or assemble final products for the Company. In addition, the Company has made prepayments associated with long-term supply agreements to secure supply of inventory components. As of September 28, 2024, the Company’s vendor non-trade receivables and prepayments related to long-term supply agreements were concentrated among a few individual vendors located primarily in Asia. While the Company has procedures to monitor and limit exposure to credit risk on its trade and vendor non-trade receivables, as well as long-term prepayments, there can be no assurance such procedures will effectively limit its credit risk and avoid losses.\nThe Company is subject to changes in tax rates, the adoption of new U.S. or international tax legislation and exposure to additional tax liabilities.\nThe Company is subject to taxes in the U.S. and numerous foreign jurisdictions, including Ireland and Singapore, where a number of the Company’s subsidiaries are organized. Due to economic and political conditions, tax laws and tax rates for income taxes and other non-income taxes in various jurisdictions may be subject to significant change. For example, the Organisation for Economic Co-operation and Development continues to advance proposals for modernizing international tax rules, including the introduction of global minimum tax standards. The Company’s effective tax rates are affected by changes in the mix of earnings in countries with differing statutory tax rates, changes in the valuation of deferred tax assets and liabilities, the introduction of new taxes, and changes in tax laws or their interpretation. The application of tax laws may be uncertain, require significant judgment and be subject to differing interpretations.\nApple Inc. | 2024 Form 10-K | 16\n\nThe Company is also subject to the examination of its tax returns and other tax matters by the U.S. Internal Revenue Service and other tax authorities and governmental bodies. The Company regularly assesses the likelihood of an adverse outcome resulting from these examinations to determine the adequacy of its provision for taxes. There can be no assurance as to the outcome of these examinations. If the Company’s effective tax rates were to increase, or if the ultimate determination of the Company’s taxes owed is for an amount in excess of amounts previously accrued, the Company’s business, results of operations and financial condition could be materially adversely affected.\nGeneral Risks\nThe price of the Company’s stock is subject to volatility.\nThe Company’s stock has experienced substantial price volatility in the past and may continue to do so in the future. Additionally, the Company, the technology industry and the stock market as a whole have, from time to time, experienced extreme stock price and volume fluctuations that have affected stock prices in ways that may have been unrelated to these companies’ operating performance. Price volatility may cause the average price at which the Company repurchases its stock in a given period to exceed the stock’s price at a given point in time. The Company believes the price of its stock should reflect expectations of future growth and profitability. The Company also believes the price of its stock should reflect expectations that its cash dividend will continue at current levels or grow, and that its current share repurchase program will be fully consummated. Future dividends are subject to declaration by the Company’s Board of Directors (the “Board”), and the Company’s share repurchase program does not obligate it to acquire any specific number of shares. If the Company fails to meet expectations related to future growth, profitability, dividends, share repurchases or other market expectations, the price of the Company’s stock may decline significantly, which could have a material adverse impact on investor confidence and employee retention.\nItem 1B.    Unresolved Staff Comments\nNone.\nItem 1C.    Cybersecurity\nThe Company’s management, led by its Head of Corporate Information Security, has overall responsibility for identifying, assessing and managing any material risks from cybersecurity threats. The Company’s Head of Corporate Information Security leads a dedicated Information Security team of highly skilled individuals with experience across industries that, among other things, develops and distributes information security policies, standards and procedures; engages in employee cybersecurity training; implements security controls; assesses security risk and compliance posture; monitors and responds to security events; and executes security testing and assessments. The Company’s Head of Corporate Information Security has extensive knowledge and skills gained from over 25 years of experience in the cybersecurity industry, including serving in leadership positions at other large technology companies and leading the Company’s Information Security team since 2016.\nThe Company’s Information Security team coordinates with teams across the Company to prevent, respond to and manage security incidents, and engages third parties, as appropriate, to assess, test or otherwise assist with aspects of its security processes and incident response. A dedicated Supplier Trust team manages information security risks the Company is exposed to through its supplier relationships. The Company has processes to log, track, address, and escalate for further assessment and report, as appropriate, cybersecurity incidents across the Company and its suppliers to senior management and the Audit and Finance Committee (the “Audit Committee”) of the Board. The Company’s enterprise risk management program is designed to identify, assess, and monitor the Company’s business risks, including financial, operational, compliance and reputational risks, and reflects management’s assessment of cybersecurity risks.\nThe Audit Committee assists the Board in the oversight and monitoring of cybersecurity matters. The Audit Committee regularly reviews and discusses the Company’s cybersecurity risks with management, including the Company’s Head of Corporate Information Security, its General Counsel and the Heads of Compliance and Business Conduct, Business Assurance, and Internal Audit, and receives updates, as necessary, regarding cybersecurity incidents. The Chair of the Audit Committee regularly reports the substance of such reviews and discussions to the Board, as necessary, and recommends to the Board such actions as the Audit Committee deems appropriate.\nFor a discussion of the Company’s cybersecurity-related risks, see Item 1A of this Form 10-K under the heading “Risk Factors.”\nApple Inc. | 2024 Form 10-K | 17\n\nItem 2.    Properties\nThe Company’s headquarters is located in Cupertino, California. As of September 28, 2024, the Company owned or leased facilities and land for corporate functions, R&D, data centers, retail and other purposes at locations throughout the U.S. and in various places outside the U.S. The Company believes its existing facilities and equipment, which are used by all reportable segments, are in good operating condition and are suitable for the conduct of its business.\nItem 3.    Legal Proceedings\nDigital Markets Act Investigations\nOn March 25, 2024, the Commission announced that it had opened two formal noncompliance investigations against the Company under the DMA. The Commission’s investigations concern (1) Article 5(4) of the DMA, which relates to how developers may communicate and promote offers to end users for apps distributed through the App Store as well as how developers may conclude contracts with those end users; and (2) Article 6(3) of the DMA, which relates to default settings, uninstallation of apps, and a web browser choice screen on iOS. On June 24, 2024, the Commission announced its preliminary findings in the Article 5(4) investigation alleging that the Company’s App Store rules are in breach of the DMA and announced that it had opened a third formal investigation against the Company regarding whether the Company’s new contractual requirements for third-party app developers and app marketplaces may violate the DMA. If the Commission makes a final determination that there has been a violation, it can issue a cease and desist order and may impose fines up to 10% of the Company’s annual worldwide net sales. Although any decision by the Commission can be appealed to the General Court of the EU, the effectiveness of the Commission’s order would apply immediately while the appeal is pending, unless a stay of the order is granted. The Company believes that it complies with the DMA and has continued to make changes to its compliance plan in response to feedback and engagement with the Commission.\nDepartment of Justice Lawsuit\nOn March 21, 2024, the U.S. Department of Justice (the “DOJ”) and a number of state and district attorneys general filed a civil antitrust lawsuit in the U.S. District Court for the District of New Jersey against the Company alleging monopolization or attempted monopolization in the markets for “performance smartphones” and “smartphones” in violation of U.S. antitrust laws. The DOJ is seeking equitable relief to redress the alleged anticompetitive behavior. In addition, various civil litigation matters have been filed in state and federal courts in the U.S. alleging similar violations of U.S. antitrust laws and seeking monetary damages and other nonmonetary relief. The Company believes it has substantial defenses and intends to vigorously defend itself.\nEpic Games\nEpic Games, Inc. (“Epic”) filed a lawsuit in the U.S. District Court for the Northern District of California (the “California District Court”) against the Company alleging violations of federal and state antitrust laws and California’s unfair competition law based upon the Company’s operation of its App Store. The California District Court found that certain provisions of the Company’s App Store Review Guidelines violate California’s unfair competition law and issued an injunction enjoining the Company from prohibiting developers from including in their apps external links that direct customers to purchasing mechanisms other than Apple in-app purchasing. The injunction applies to apps on the U.S. storefront of the iOS and iPadOS App Store. On January 16, 2024, the Company implemented a plan to comply with the injunction and filed a statement of compliance with the California District Court. A motion by Epic disputing the Company’s compliance plan and seeking to enforce the injunction, which the Company has opposed, is pending before the California District Court. On September 30, 2024, the Company filed a motion with the California District Court to narrow or vacate the injunction. The Company believes it has substantial defenses and intends to vigorously defend itself.\nOther Legal Proceedings\nThe Company is subject to other legal proceedings and claims that have not been fully resolved and that have arisen in the ordinary course of business. The Company settled certain matters during the fourth quarter of 2024 that did not individually or in the aggregate have a material impact on the Company’s financial condition or operating results. The outcome of litigation is inherently uncertain. If one or more legal matters were resolved against the Company in a reporting period for amounts above management’s expectations, the Company’s financial condition and operating results for that reporting period could be materially adversely affected.\nItem 4.    Mine Safety Disclosures\nNot applicable.\nApple Inc. | 2024 Form 10-K | 18\n\nPART II\nItem 5.    Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities\nThe Company’s common stock is traded on The Nasdaq Stock Market LLC under the symbol AAPL.\nHolders\nAs of October 18, 2024, there were 23,301 shareholders of record.\nPurchases of Equity Securities by the Issuer and Affiliated Purchasers\nShare repurchase activity during the three months ended September 28, 2024 was as follows (in millions, except number of shares, which are reflected in thousands, and per-share amounts):\nPeriods\t\tTotal Number\nof Shares Purchased\t\tAverage Price\nPaid Per Share\t\tTotal Number of Shares\nPurchased as Part of Publicly\nAnnounced Plans or Programs\t\t\nApproximate Dollar Value of\nShares That May Yet Be Purchased\nUnder the Plans or Programs (1)\nJune 30, 2024 to August 3, 2024:\nOpen market and privately negotiated purchases\t\t35,697 \t\t\t$\t224.11 \t\t\t35,697 \t\t\t\nAugust 4, 2024 to August 31, 2024:\nOpen market and privately negotiated purchases\t\t42,910 \t\t\t$\t221.39 \t\t\t42,910 \t\t\t\nSeptember 1, 2024 to September 28, 2024:\nOpen market and privately negotiated purchases\t\t33,653 \t\t\t$\t222.86 \t\t\t33,653 \t\t\t\nTotal\t\t112,260 \t\t\t\t\t\t\t$\t89,074 \t\n \n(1)As of September 28, 2024, the Company was authorized by the Board to purchase up to $110 billion of the Company’s common stock under a share repurchase program announced on May 2, 2024, of which $20.9 billion had been utilized. During the fourth quarter of 2024, the Company also utilized the final $4.1 billion under its previous repurchase program, which was authorized in May 2023. The programs do not obligate the Company to acquire a minimum amount of shares. Under the programs, shares may be repurchased in privately negotiated or open market transactions, including under plans complying with Rule 10b5-1 under the Exchange Act.\nApple Inc. | 2024 Form 10-K | 19\n\nCompany Stock Performance\nThe following graph shows a comparison of five-year cumulative total shareholder return, calculated on a dividend-reinvested basis, for the Company, the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index. The graph assumes $100 was invested in each of the Company’s common stock, the S&P 500 Index and the Dow Jones U.S. Technology Supersector Index as of the market close on September 27, 2019. Past stock price performance is not necessarily indicative of future stock price performance.\n2218\nSeptember 2019\t\tSeptember 2020\t\tSeptember 2021\t\tSeptember 2022\t\tSeptember 2023\t\tSeptember 2024\nApple Inc.\t\t$\t100 \t\t\t$\t207 \t\t\t$\t273 \t\t\t$\t281 \t\t\t$\t322 \t\t\t$\t430 \t\nS&P 500 Index\t\t$\t100 \t\t\t$\t113 \t\t\t$\t156 \t\t\t$\t131 \t\t\t$\t155 \t\t\t$\t210 \t\nDow Jones U.S. Technology Supersector Index\t\t$\t100 \t\t\t$\t146 \t\t\t$\t216 \t\t\t$\t156 \t\t\t$\t215 \t\t\t$\t322 \t\n \nItem 6.    [Reserved]\nApple Inc. | 2024 Form 10-K | 20\n\nItem 7.    Management’s Discussion and Analysis of Financial Condition and Results of Operations\nThe following discussion should be read in conjunction with the consolidated financial statements and accompanying notes included in Part II, Item 8 of this Form 10-K. This Item generally discusses 2024 and 2023 items and year-to-year comparisons between 2024 and 2023. Discussions of 2022 items and year-to-year comparisons between 2023 and 2022 are not included, and can be found in “Management’s Discussion and Analysis of Financial Condition and Results of Operations” in Part II, Item 7 of the Company’s Annual Report on Form 10-K for the fiscal year ended September 30, 2023.\nProduct, Service and Software Announcements\nThe Company announces new product, service and software offerings at various times during the year. Significant announcements during fiscal year 2024 included the following:\nFirst Quarter 2024:\n•MacBook Pro 14-in.;\n•MacBook Pro 16-in.; and\n•iMac.\nSecond Quarter 2024:\n•MacBook Air 13-in.; and\n•MacBook Air 15-in.\nThird Quarter 2024:\n•iPad Air;\n•iPad Pro;\n•iOS 18, macOS Sequoia, iPadOS 18, watchOS 11, visionOS 2 and tvOS 18, updates to the Company’s operating systems; and\n•Apple Intelligence™, a personal intelligence system that uses generative models.\nFourth Quarter 2024:\n•iPhone 16, iPhone 16 Plus, iPhone 16 Pro and iPhone 16 Pro Max;\n•Apple Watch Series 10; and\n•AirPods 4.\nFiscal Period\nThe Company’s fiscal year is the 52- or 53-week period that ends on the last Saturday of September. An additional week is included in the first fiscal quarter every five or six years to realign the Company’s fiscal quarters with calendar quarters, which occurred in the first quarter of 2023. The Company’s fiscal years 2024 and 2022 spanned 52 weeks each, whereas fiscal year 2023 spanned 53 weeks.\nMacroeconomic Conditions\nMacroeconomic conditions, including inflation, interest rates and currency fluctuations, have directly and indirectly impacted, and could in the future materially impact, the Company’s results of operations and financial condition.\nApple Inc. | 2024 Form 10-K | 21\n\nSegment Operating Performance\nThe following table shows net sales by reportable segment for 2024, 2023 and 2022 (dollars in millions):\n2024\t\tChange\t\t2023\t\tChange\t\t2022\nAmericas\t$\t167,045 \t\t\t3 \t%\t\t$\t162,560 \t\t\t(4)\t%\t\t$\t169,658 \t\nEurope\t101,328 \t\t\t7 \t%\t\t94,294 \t\t\t(1)\t%\t\t95,118 \t\nGreater China\t66,952 \t\t\t(8)\t%\t\t72,559 \t\t\t(2)\t%\t\t74,200 \t\nJapan\t25,052 \t\t\t3 \t%\t\t24,257 \t\t\t(7)\t%\t\t25,977 \t\nRest of Asia Pacific\t30,658 \t\t\t4 \t%\t\t29,615 \t\t\t1 \t%\t\t29,375 \t\nTotal net sales\t$\t391,035 \t\t\t2 \t%\t\t$\t383,285 \t\t\t(3)\t%\t\t$\t394,328 \t\n \nAmericas\nAmericas net sales increased during 2024 compared to 2023 due primarily to higher net sales of Services.\nEurope\nEurope net sales increased during 2024 compared to 2023 due primarily to higher net sales of Services and iPhone.\nGreater China\nGreater China net sales decreased during 2024 compared to 2023 due primarily to lower net sales of iPhone and iPad. The weakness in the renminbi relative to the U.S. dollar had an unfavorable year-over-year impact on Greater China net sales during 2024.\nJapan\nJapan net sales increased during 2024 compared to 2023 due primarily to higher net sales of iPhone. The weakness in the yen relative to the U.S. dollar had an unfavorable year-over-year impact on Japan net sales during 2024.\nRest of Asia Pacific\nRest of Asia Pacific net sales increased during 2024 compared to 2023 due primarily to higher net sales of Services. The weakness in foreign currencies relative to the U.S. dollar had a net unfavorable year-over-year impact on Rest of Asia Pacific net sales during 2024.\nApple Inc. | 2024 Form 10-K | 22\n\nProducts and Services Performance\nThe following table shows net sales by category for 2024, 2023 and 2022 (dollars in millions):\n2024\t\tChange\t\t2023\t\tChange\t\t2022\niPhone\t$\t201,183 \t\t\t— \t%\t\t$\t200,583 \t\t\t(2)\t%\t\t$\t205,489 \t\nMac\t29,984 \t\t\t2 \t%\t\t29,357 \t\t\t(27)\t%\t\t40,177 \t\niPad\t26,694 \t\t\t(6)\t%\t\t28,300 \t\t\t(3)\t%\t\t29,292 \t\nWearables, Home and Accessories\t37,005 \t\t\t(7)\t%\t\t39,845 \t\t\t(3)\t%\t\t41,241 \t\nServices (1)\n96,169 \t\t\t13 \t%\t\t85,200 \t\t\t9 \t%\t\t78,129 \t\nTotal net sales\t$\t391,035 \t\t\t2 \t%\t\t$\t383,285 \t\t\t(3)\t%\t\t$\t394,328 \t\n \n(1)Services net sales include amortization of the deferred value of services bundled in the sales price of certain products.\niPhone\niPhone net sales were relatively flat during 2024 compared to 2023.\nMac\nMac net sales increased during 2024 compared to 2023 due primarily to higher net sales of laptops.\niPad\niPad net sales decreased during 2024 compared to 2023 due primarily to lower net sales of iPad Pro and the entry-level iPad models, partially offset by higher net sales of iPad Air.\nWearables, Home and Accessories\nWearables, Home and Accessories net sales decreased during 2024 compared to 2023 due primarily to lower net sales of Wearables and Accessories.\nServices\nServices net sales increased during 2024 compared to 2023 due primarily to higher net sales from advertising, the App Store® and cloud services.\nApple Inc. | 2024 Form 10-K | 23\n\nGross Margin\nProducts and Services gross margin and gross margin percentage for 2024, 2023 and 2022 were as follows (dollars in millions):\n2024\t\t2023\t\t2022\nGross margin:\t\t\t\t\t\nProducts\t$\t109,633 \t\t\t$\t108,803 \t\t\t$\t114,728 \t\nServices\t71,050 \t\t\t60,345 \t\t\t56,054 \t\nTotal gross margin\t$\t180,683 \t\t\t$\t169,148 \t\t\t$\t170,782 \t\n \nGross margin percentage:\t\t\t\t\t\nProducts\t37.2 \t%\t\t36.5 \t%\t\t36.3 \t%\nServices\t73.9 \t%\t\t70.8 \t%\t\t71.7 \t%\nTotal gross margin percentage\t46.2 \t%\t\t44.1 \t%\t\t43.3 \t%\n \nProducts Gross Margin\nProducts gross margin and Products gross margin percentage increased during 2024 compared to 2023 due to cost savings, partially offset by a different Products mix and the weakness in foreign currencies relative to the U.S. dollar.\nServices Gross Margin\nServices gross margin increased during 2024 compared to 2023 due primarily to higher Services net sales.\nServices gross margin percentage increased during 2024 compared to 2023 due to a different Services mix.\nThe Company’s future gross margins can be impacted by a variety of factors, as discussed in Part I, Item 1A of this Form 10-K under the heading “Risk Factors.” As a result, the Company believes, in general, gross margins will be subject to volatility and downward pressure.\nOperating Expenses\nOperating expenses for 2024, 2023 and 2022 were as follows (dollars in millions):\n2024\t\tChange\t\t2023\t\tChange\t\t2022\nResearch and development\t$\t31,370 \t\t\t5 \t%\t\t$\t29,915 \t\t\t14 \t%\t\t$\t26,251 \t\nPercentage of total net sales\t8 \t%\t\t\t\t8 \t%\t\t\t\t7 \t%\nSelling, general and administrative\t$\t26,097 \t\t\t5 \t%\t\t$\t24,932 \t\t\t(1)\t%\t\t$\t25,094 \t\nPercentage of total net sales\t7 \t%\t\t\t\t7 \t%\t\t\t\t6 \t%\nTotal operating expenses\t$\t57,467 \t\t\t5 \t%\t\t$\t54,847 \t\t\t7 \t%\t\t$\t51,345 \t\nPercentage of total net sales\t15 \t%\t\t\t\t14 \t%\t\t\t\t13 \t%\n \nResearch and Development\nThe growth in R&D expense during 2024 compared to 2023 was driven primarily by increases in headcount-related expenses.\nSelling, General and Administrative\nSelling, general and administrative expense increased $1.2 billion during 2024 compared to 2023.\nApple Inc. | 2024 Form 10-K | 24\n\nProvision for Income Taxes\nProvision for income taxes, effective tax rate and statutory federal income tax rate for 2024, 2023 and 2022 were as follows (dollars in millions):\n2024\t\t2023\t\t2022\nProvision for income taxes\t$\t29,749 \t\t\t$\t16,741 \t\t\t$\t19,300 \t\nEffective tax rate\t24.1 \t%\t\t14.7 \t%\t\t16.2 \t%\nStatutory federal income tax rate\t21 \t%\t\t21 \t%\t\t21 \t%\n \nThe Company’s effective tax rate for 2024 was higher than the statutory federal income tax rate due primarily to a one-time income tax charge of $10.2 billion, net, related to the State Aid Decision (refer to Note 7, “Income Taxes” in the Notes to Consolidated Financial Statements in Part II, Item 8 of this Form 10-K) and state income taxes, partially offset by a lower effective tax rate on foreign earnings, the impact of the U.S. federal R&D credit, and tax benefits from share-based compensation.\nThe Company’s effective tax rate for 2024 was higher compared to 2023 due primarily to a one-time income tax charge of $10.2 billion, net, related to the State Aid Decision, a higher effective tax rate on foreign earnings and lower tax benefits from share-based compensation.\nLiquidity and Capital Resources\nThe Company believes its balances of unrestricted cash, cash equivalents and marketable securities, which totaled $140.8 billion as of September 28, 2024, along with cash generated by ongoing operations and continued access to debt markets, will be sufficient to satisfy its cash requirements and capital return program over the next 12 months and beyond.\nThe Company’s material cash requirements include the following contractual obligations:\nDebt\nAs of September 28, 2024, the Company had outstanding fixed-rate notes with varying maturities for an aggregate principal amount of $97.3 billion (collectively the “Notes”), with $10.9 billion payable within 12 months. Future interest payments associated with the Notes total $38.5 billion, with $2.6 billion payable within 12 months.\nThe Company also issues unsecured short-term promissory notes pursuant to a commercial paper program. As of September 28, 2024, the Company had $10.0 billion of commercial paper outstanding, all of which was payable within 12 months.\nLeases\nThe Company has lease arrangements for certain equipment and facilities, including corporate, data center, manufacturing and retail space. As of September 28, 2024, the Company had fixed lease payment obligations of $15.6 billion, with $2.0 billion payable within 12 months.\nManufacturing Purchase Obligations\nThe Company utilizes several outsourcing partners to manufacture subassemblies for the Company’s products and to perform final assembly and testing of finished products. The Company also obtains individual components for its products from a wide variety of individual suppliers. As of September 28, 2024, the Company had manufacturing purchase obligations of $53.0 billion, with $52.9 billion payable within 12 months.\nOther Purchase Obligations\nThe Company’s other purchase obligations primarily consist of noncancelable obligations to acquire capital assets, including assets related to product manufacturing, and noncancelable obligations related to supplier arrangements, licensed intellectual property and content, and distribution rights. As of September 28, 2024, the Company had other purchase obligations of $12.0 billion, with $4.1 billion payable within 12 months.\nDeemed Repatriation Tax Payable\nAs of September 28, 2024, the balance of the deemed repatriation tax payable imposed by the U.S. Tax Cuts and Jobs Act of 2017 (the “TCJA”) was $16.5 billion, with $7.2 billion expected to be paid within 12 months.\nApple Inc. | 2024 Form 10-K | 25\n\nState Aid Decision Tax Payable\nAs of September 28, 2024, the Company had an obligation to pay €14.2 billion or $15.8 billion to Ireland in connection with the State Aid Decision, all of which was expected to be paid within 12 months. The funds necessary to settle the obligation were held in escrow as of September 28, 2024, and restricted from general use.\nCapital Return Program\nIn addition to its contractual cash requirements, the Company has an authorized share repurchase program. The program does not obligate the Company to acquire a minimum amount of shares. As of September 28, 2024, the Company’s quarterly cash dividend was $0.25 per share. The Company intends to increase its dividend on an annual basis, subject to declaration by the Board.\nIn May 2024, the Company announced a new share repurchase program of up to $110 billion and raised its quarterly dividend from $0.24 to $0.25 per share beginning in May 2024. During 2024, the Company repurchased $95.0 billion of its common stock and paid dividends and dividend equivalents of $15.2 billion.\nRecent Accounting Pronouncements\nIncome Taxes\nIn December 2023, the Financial Accounting Standards Board (the “FASB”) issued Accounting Standards Update (“ASU”) No. 2023-09, Income Taxes (Topic 740): Improvements to Income Tax Disclosures (“ASU 2023-09”), which will require the Company to disclose specified additional information in its income tax rate reconciliation and provide additional information for reconciling items that meet a quantitative threshold. ASU 2023-09 will also require the Company to disaggregate its income taxes paid disclosure by federal, state and foreign taxes, with further disaggregation required for significant individual jurisdictions. The Company will adopt ASU 2023-09 in its fourth quarter of 2026 using a prospective transition method.\nSegment Reporting\nIn November 2023, the FASB issued ASU No. 2023-07, Segment Reporting (Topic 280): Improvements to Reportable Segment Disclosures (“ASU 2023-07”), which will require the Company to disclose segment expenses that are significant and regularly provided to the Company’s chief operating decision maker (“CODM”). In addition, ASU 2023-07 will require the Company to disclose the title and position of its CODM and how the CODM uses segment profit or loss information in assessing segment performance and deciding how to allocate resources. The Company will adopt ASU 2023-07 in its fourth quarter of 2025 using a retrospective transition method.\nCritical Accounting Estimates\nThe preparation of financial statements and related disclosures in conformity with U.S. generally accepted accounting principles (“GAAP”) and the Company’s discussion and analysis of its financial condition and operating results require the Company’s management to make judgments, assumptions and estimates that affect the amounts reported. Note 1, “Summary of Significant Accounting Policies” of the Notes to Consolidated Financial Statements in Part II, Item 8 of this Form 10-K describes the significant accounting policies and methods used in the preparation of the Company’s consolidated financial statements. Management bases its estimates on historical experience and on various other assumptions it believes to be reasonable under the circumstances, the results of which form the basis for making judgments about the carrying values of assets and liabilities.\nUncertain Tax Positions\nThe Company is subject to income taxes in the U.S. and numerous foreign jurisdictions. The evaluation of the Company’s uncertain tax positions involves significant judgment in the interpretation and application of GAAP and complex domestic and international tax laws, including the TCJA and the allocation of international taxation rights between countries. Although management believes the Company’s reserves are reasonable, no assurance can be given that the final outcome of these uncertainties will not be different from that reflected in the Company’s reserves. Reserves are adjusted considering changing facts and circumstances, such as the closing of a tax examination. Resolution of these uncertainties in a manner inconsistent with management’s expectations could have a material impact on the Company’s financial condition and operating results.\nLegal and Other Contingencies\nThe Company is subject to various legal proceedings and claims that arise in the ordinary course of business, the outcomes of which are inherently uncertain. The Company records a liability when it is probable that a loss has been incurred and the amount is reasonably estimable, the determination of which requires significant judgment. Resolution of legal matters in a manner inconsistent with management’s expectations could have a material impact on the Company’s financial condition and operating results.\nApple Inc. | 2024 Form 10-K | 26\n\nItem 7A.    Quantitative and Qualitative Disclosures About Market Risk\nThe Company is exposed to economic risk from interest rates and foreign exchange rates. The Company uses various strategies to manage these risks; however, they may still impact the Company’s consolidated financial statements.\nInterest Rate Risk\nThe Company is primarily exposed to fluctuations in U.S. interest rates and their impact on the Company’s investment portfolio and term debt. Increases in interest rates will negatively affect the fair value of the Company’s investment portfolio and increase the interest expense on the Company’s term debt. To protect against interest rate risk, the Company may use derivative instruments, offset interest rate–sensitive assets and liabilities, or control duration of the investment and term debt portfolios.\nThe following table sets forth potential impacts on the Company’s investment portfolio and term debt, including the effects of any associated derivatives, that would result from a hypothetical increase in relevant interest rates as of September 28, 2024 and September 30, 2023 (dollars in millions):\nInterest Rate\nSensitive Instrument\nHypothetical Interest\nRate Increase\nPotential Impact\n2024\t\t2023\nInvestment portfolio\n100 basis points, all tenors\nDecline in fair value\n$\t2,755 \t\t\t$\t3,089 \t\nTerm debt\n100 basis points, all tenors\nIncrease in annual interest expense\n$\t139 \t\t\t$\t194 \t\n \nForeign Exchange Rate Risk\nThe Company’s exposure to foreign exchange rate risk relates primarily to the Company being a net receiver of currencies other than the U.S. dollar. Changes in exchange rates, and in particular a strengthening of the U.S. dollar, will negatively affect the Company’s net sales and gross margins as expressed in U.S. dollars. Fluctuations in exchange rates may also affect the fair values of certain of the Company’s assets and liabilities. To protect against foreign exchange rate risk, the Company may use derivative instruments, offset exposures, or adjust local currency pricing of its products and services. However, the Company may choose to not hedge certain foreign currency exposures for a variety of reasons, including accounting considerations or prohibitive cost.\nThe Company applied a value-at-risk (“VAR”) model to its foreign currency derivative positions to assess the potential impact of fluctuations in exchange rates. The VAR model used a Monte Carlo simulation. The VAR is the maximum expected loss in fair value, for a given confidence interval, to the Company’s foreign currency derivative positions due to adverse movements in rates. Based on the results of the model, the Company estimates, with 95% confidence, a maximum one-day loss in fair value of $538 million and $669 million as of September 28, 2024 and September 30, 2023, respectively. Changes in the Company’s underlying foreign currency exposures, which were excluded from the assessment, generally offset changes in the fair values of the Company’s foreign currency derivatives.\nApple Inc. | 2024 Form 10-K | 27\n\nItem 8.    Financial Statements and Supplementary Data\nIndex to Consolidated Financial Statements\t\tPage\nConsolidated Statements of Operations for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n29\nConsolidated Statements of Comprehensive Income for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n30\nConsolidated Balance Sheets as of September 28, 2024 and September 30, 2023\n31\nConsolidated Statements of Shareholders’ Equity for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n32\nConsolidated Statements of Cash Flows for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n33\nNotes to Consolidated Financial Statements\n34\nReports of Independent Registered Public Accounting Firm\n48\n \nAll financial statement schedules have been omitted, since the required information is not applicable or is not present in amounts sufficient to require submission of the schedule, or because the information required is included in the consolidated financial statements and accompanying notes.\nApple Inc. | 2024 Form 10-K | 28\n\nApple Inc.\nCONSOLIDATED STATEMENTS OF OPERATIONS\n(In millions, except number of shares, which are reflected in thousands, and per-share amounts)\n\nYears ended\nSeptember 28,\n2024\t\tSeptember 30,\n2023\t\tSeptember 24,\n2022\nNet sales:\t\t\t\t\t\n   Products\t$\t294,866 \t\t\t$\t298,085 \t\t\t$\t316,199 \t\n   Services\t96,169 \t\t\t85,200 \t\t\t78,129 \t\nTotal net sales\t391,035 \t\t\t383,285 \t\t\t394,328 \t\nCost of sales:\t\t\t\t\t\n   Products\t185,233 \t\t\t189,282 \t\t\t201,471 \t\n   Services\t25,119 \t\t\t24,855 \t\t\t22,075 \t\nTotal cost of sales\t210,352 \t\t\t214,137 \t\t\t223,546 \t\nGross margin\t180,683 \t\t\t169,148 \t\t\t170,782 \t\nOperating expenses:\t\t\t\t\t\nResearch and development\t31,370 \t\t\t29,915 \t\t\t26,251 \t\nSelling, general and administrative\t26,097 \t\t\t24,932 \t\t\t25,094 \t\nTotal operating expenses\t57,467 \t\t\t54,847 \t\t\t51,345 \t\nOperating income\t123,216 \t\t\t114,301 \t\t\t119,437 \t\nOther income/(expense), net\t269 \t\t\t(565)\t\t\t(334)\t\nIncome before provision for income taxes\t123,485 \t\t\t113,736 \t\t\t119,103 \t\nProvision for income taxes\t29,749 \t\t\t16,741 \t\t\t19,300 \t\nNet income\t$\t93,736 \t\t\t$\t96,995 \t\t\t$\t99,803 \t\nEarnings per share:\t\t\t\t\t\nBasic\t$\t6.11 \t\t\t$\t6.16 \t\t\t$\t6.15 \t\nDiluted\t$\t6.08 \t\t\t$\t6.13 \t\t\t$\t6.11 \t\nShares used in computing earnings per share:\t\t\t\t\t\nBasic\t15,343,783 \t\t\t15,744,231 \t\t\t16,215,963 \t\nDiluted\t15,408,095 \t\t\t15,812,547 \t\t\t16,325,819 \t\n \nSee accompanying Notes to Consolidated Financial Statements.\nApple Inc. | 2024 Form 10-K | 29\n\nApple Inc.\nCONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME\n(In millions)\n\nYears ended\nSeptember 28,\n2024\t\tSeptember 30,\n2023\t\tSeptember 24,\n2022\nNet income\t$\t93,736 \t\t\t$\t96,995 \t\t\t$\t99,803 \t\nOther comprehensive income/(loss):\t\t\t\t\t\nChange in foreign currency translation, net of tax\t395 \t\t\t(765)\t\t\t(1,511)\t\nChange in unrealized gains/losses on derivative instruments, net of tax:\t\t\t\t\t\nChange in fair value of derivative instruments\t(832)\t\t\t323 \t\t\t3,212 \t\nAdjustment for net (gains)/losses realized and included in net income\t(1,337)\t\t\t(1,717)\t\t\t(1,074)\t\nTotal change in unrealized gains/losses on derivative instruments\t(2,169)\t\t\t(1,394)\t\t\t2,138 \t\nChange in unrealized gains/losses on marketable debt securities, net of tax:\t\t\t\t\t\nChange in fair value of marketable debt securities\t5,850 \t\t\t1,563 \t\t\t(12,104)\t\nAdjustment for net (gains)/losses realized and included in net income\t204 \t\t\t253 \t\t\t205 \t\nTotal change in unrealized gains/losses on marketable debt securities\t6,054 \t\t\t1,816 \t\t\t(11,899)\t\nTotal other comprehensive income/(loss)\t4,280 \t\t\t(343)\t\t\t(11,272)\t\nTotal comprehensive income\t$\t98,016 \t\t\t$\t96,652 \t\t\t$\t88,531 \t\n \nSee accompanying Notes to Consolidated Financial Statements.\nApple Inc. | 2024 Form 10-K | 30\n\nApple Inc.\nCONSOLIDATED BALANCE SHEETS\n(In millions, except number of shares, which are reflected in thousands, and par value)\n\nSeptember 28,\n2024\t\tSeptember 30,\n2023\nASSETS:\nCurrent assets:\t\t\t\nCash and cash equivalents\t$\t29,943 \t\t\t$\t29,965 \t\nMarketable securities\t35,228 \t\t\t31,590 \t\nAccounts receivable, net\t33,410 \t\t\t29,508 \t\nVendor non-trade receivables\t32,833 \t\t\t31,477 \t\nInventories\t7,286 \t\t\t6,331 \t\nOther current assets\t14,287 \t\t\t14,695 \t\nTotal current assets\t152,987 \t\t\t143,566 \t\nNon-current assets:\t\t\t\nMarketable securities\t91,479 \t\t\t100,544 \t\nProperty, plant and equipment, net\t45,680 \t\t\t43,715 \t\nOther non-current assets\t74,834 \t\t\t64,758 \t\nTotal non-current assets\t211,993 \t\t\t209,017 \t\nTotal assets\t$\t364,980 \t\t\t$\t352,583 \t\nLIABILITIES AND SHAREHOLDERS’ EQUITY:\nCurrent liabilities:\t\t\t\nAccounts payable\t$\t68,960 \t\t\t$\t62,611 \t\nOther current liabilities\t78,304 \t\t\t58,829 \t\nDeferred revenue\t8,249 \t\t\t8,061 \t\nCommercial paper\t9,967 \t\t\t5,985 \t\nTerm debt\t10,912 \t\t\t9,822 \t\nTotal current liabilities\t176,392 \t\t\t145,308 \t\nNon-current liabilities:\t\t\t\nTerm debt\t85,750 \t\t\t95,281 \t\nOther non-current liabilities\t45,888 \t\t\t49,848 \t\nTotal non-current liabilities\t131,638 \t\t\t145,129 \t\nTotal liabilities\t308,030 \t\t\t290,437 \t\nCommitments and contingencies\t\t\t\nShareholders’ equity:\t\t\t\nCommon stock and additional paid-in capital, $0.00001 par value: 50,400,000 shares authorized; 15,116,786 and 15,550,061 shares issued and outstanding, respectively\n83,276 \t\t\t73,812 \t\nAccumulated deficit\t(19,154)\t\t\t(214)\t\nAccumulated other comprehensive loss\t(7,172)\t\t\t(11,452)\t\nTotal shareholders’ equity\t56,950 \t\t\t62,146 \t\nTotal liabilities and shareholders’ equity\t$\t364,980 \t\t\t$\t352,583 \t\n \nSee accompanying Notes to Consolidated Financial Statements.\nApple Inc. | 2024 Form 10-K | 31\n\nApple Inc.\nCONSOLIDATED STATEMENTS OF SHAREHOLDERS’ EQUITY\n(In millions, except per-share amounts)\n\nYears ended\nSeptember 28,\n2024\t\tSeptember 30,\n2023\t\tSeptember 24,\n2022\nTotal shareholders’ equity, beginning balances\t$\t62,146 \t\t\t$\t50,672 \t\t\t$\t63,090 \t\nCommon stock and additional paid-in capital:\t\t\t\t\t\nBeginning balances\t73,812 \t\t\t64,849 \t\t\t57,365 \t\nCommon stock issued\t1,423 \t\t\t1,346 \t\t\t1,175 \t\nCommon stock withheld related to net share settlement of equity awards\t(3,993)\t\t\t(3,521)\t\t\t(2,971)\t\nShare-based compensation\t12,034 \t\t\t11,138 \t\t\t9,280 \t\nEnding balances\t83,276 \t\t\t73,812 \t\t\t64,849 \t\nRetained earnings/(Accumulated deficit):\t\t\t\t\t\nBeginning balances\t(214)\t\t\t(3,068)\t\t\t5,562 \t\nNet income\t93,736 \t\t\t96,995 \t\t\t99,803 \t\nDividends and dividend equivalents declared\t(15,218)\t\t\t(14,996)\t\t\t(14,793)\t\nCommon stock withheld related to net share settlement of equity awards\t(1,612)\t\t\t(2,099)\t\t\t(3,454)\t\nCommon stock repurchased\t(95,846)\t\t\t(77,046)\t\t\t(90,186)\t\nEnding balances\t(19,154)\t\t\t(214)\t\t\t(3,068)\t\nAccumulated other comprehensive income/(loss):\t\t\t\t\t\nBeginning balances\t(11,452)\t\t\t(11,109)\t\t\t163 \t\nOther comprehensive income/(loss)\t4,280 \t\t\t(343)\t\t\t(11,272)\t\nEnding balances\t(7,172)\t\t\t(11,452)\t\t\t(11,109)\t\nTotal shareholders’ equity, ending balances\t$\t56,950 \t\t\t$\t62,146 \t\t\t$\t50,672 \t\nDividends and dividend equivalents declared per share or RSU\t$\t0.98 \t\t\t$\t0.94 \t\t\t$\t0.90 \t\n \nSee accompanying Notes to Consolidated Financial Statements.\nApple Inc. | 2024 Form 10-K | 32\n\nApple Inc.\nCONSOLIDATED STATEMENTS OF CASH FLOWS\n(In millions)\nYears ended\nSeptember 28,\n2024\t\tSeptember 30,\n2023\t\tSeptember 24,\n2022\nCash, cash equivalents, and restricted cash and cash equivalents, beginning balances\n$\t30,737 \t\t\t$\t24,977 \t\t\t$\t35,929 \t\nOperating activities:\t\t\t\t\t\nNet income\t93,736 \t\t\t96,995 \t\t\t99,803 \t\nAdjustments to reconcile net income to cash generated by operating activities:\t\t\t\t\t\nDepreciation and amortization\t11,445 \t\t\t11,519 \t\t\t11,104 \t\nShare-based compensation expense\t11,688 \t\t\t10,833 \t\t\t9,038 \t\nOther\t(2,266)\t\t\t(2,227)\t\t\t1,006 \t\nChanges in operating assets and liabilities:\t\t\t\t\t\nAccounts receivable, net\t(3,788)\t\t\t(1,688)\t\t\t(1,823)\t\nVendor non-trade receivables\t(1,356)\t\t\t1,271 \t\t\t(7,520)\t\nInventories\t(1,046)\t\t\t(1,618)\t\t\t1,484 \t\nOther current and non-current assets\t(11,731)\t\t\t(5,684)\t\t\t(6,499)\t\nAccounts payable\t6,020 \t\t\t(1,889)\t\t\t9,448 \t\nOther current and non-current liabilities\t15,552 \t\t\t3,031 \t\t\t6,110 \t\nCash generated by operating activities\t118,254 \t\t\t110,543 \t\t\t122,151 \t\nInvesting activities:\t\t\t\t\t\nPurchases of marketable securities\t(48,656)\t\t\t(29,513)\t\t\t(76,923)\t\nProceeds from maturities of marketable securities\t51,211 \t\t\t39,686 \t\t\t29,917 \t\nProceeds from sales of marketable securities\t11,135 \t\t\t5,828 \t\t\t37,446 \t\nPayments for acquisition of property, plant and equipment\t(9,447)\t\t\t(10,959)\t\t\t(10,708)\t\nOther\t(1,308)\t\t\t(1,337)\t\t\t(2,086)\t\nCash generated by/(used in) investing activities\t2,935 \t\t\t3,705 \t\t\t(22,354)\t\nFinancing activities:\t\t\t\t\t\nPayments for taxes related to net share settlement of equity awards\t(5,441)\t\t\t(5,431)\t\t\t(6,223)\t\nPayments for dividends and dividend equivalents\t(15,234)\t\t\t(15,025)\t\t\t(14,841)\t\nRepurchases of common stock\t(94,949)\t\t\t(77,550)\t\t\t(89,402)\t\nProceeds from issuance of term debt, net\t— \t\t\t5,228 \t\t\t5,465 \t\nRepayments of term debt\t(9,958)\t\t\t(11,151)\t\t\t(9,543)\t\nProceeds from/(Repayments of) commercial paper, net\t3,960 \t\t\t(3,978)\t\t\t3,955 \t\nOther\t(361)\t\t\t(581)\t\t\t(160)\t\nCash used in financing activities\t(121,983)\t\t\t(108,488)\t\t\t(110,749)\t\nIncrease/(Decrease) in cash, cash equivalents, and restricted cash and cash equivalents\t(794)\t\t\t5,760 \t\t\t(10,952)\t\nCash, cash equivalents, and restricted cash and cash equivalents, ending balances\n$\t29,943 \t\t\t$\t30,737 \t\t\t$\t24,977 \t\nSupplemental cash flow disclosure:\t\t\t\t\t\nCash paid for income taxes, net\t$\t26,102 \t\t\t$\t18,679 \t\t\t$\t19,573 \t\n \nSee accompanying Notes to Consolidated Financial Statements.\nApple Inc. | 2024 Form 10-K | 33\n\nApple Inc.\nNotes to Consolidated Financial Statements\nNote 1 – Summary of Significant Accounting Policies\nBasis of Presentation and Preparation\nThe consolidated financial statements include the accounts of Apple Inc. and its wholly owned subsidiaries. The preparation of these consolidated financial statements and accompanying notes in conformity with GAAP requires the use of management estimates. Certain prior period amounts in the consolidated financial statements and accompanying notes have been reclassified to conform to the current period’s presentation.\nThe Company’s fiscal year is the 52- or 53-week period that ends on the last Saturday of September. An additional week is included in the first fiscal quarter every five or six years to realign the Company’s fiscal quarters with calendar quarters, which occurred in the first fiscal quarter of 2023. The Company’s fiscal years 2024 and 2022 spanned 52 weeks each, whereas fiscal year 2023 spanned 53 weeks. Unless otherwise stated, references to particular years, quarters, months and periods refer to the Company’s fiscal years ended in September and the associated quarters, months and periods of those fiscal years.\nRevenue\nThe Company records revenue net of taxes collected from customers that are remitted to governmental authorities.\nShare-Based Compensation\nThe Company recognizes share-based compensation expense on a straight-line basis for its estimate of equity awards that will ultimately vest.\nCash Equivalents\nAll highly liquid investments with maturities of three months or less at the date of purchase are treated as cash equivalents.\nMarketable Securities\nThe cost of securities sold is determined using the specific identification method.\nInventories\nInventories are measured using the first-in, first-out method.\nProperty, Plant and Equipment\nDepreciation on property, plant and equipment is recognized on a straight-line basis.\nDerivative Instruments\nThe Company presents derivative assets and liabilities at their gross fair values in the Consolidated Balance Sheets.\nIncome Taxes\nThe Company records certain deferred tax assets and liabilities in connection with the minimum tax on certain foreign earnings created by the TCJA.\nLeases\nThe Company combines and accounts for lease and nonlease components as a single lease component for leases of corporate, data center and retail facilities.\nApple Inc. | 2024 Form 10-K | 34\n\nNote 2 – Revenue\nThe Company recognizes revenue at the amount to which it expects to be entitled when control of the products or services is transferred to its customers. Control is generally transferred when the Company has a present right to payment and title and the significant risks and rewards of ownership of products or services are transferred to its customers. For most of the Company’s Products net sales, control transfers when products are shipped. For the Company’s Services net sales, control transfers over time as services are delivered. Payment for Products and Services net sales is collected within a short period following transfer of control or commencement of delivery of services, as applicable.\nThe Company records reductions to Products net sales related to future product returns, price protection and other customer incentive programs based on the Company’s expectations and historical experience.\nFor arrangements with multiple performance obligations, which represent promises within an arrangement that are distinct, the Company allocates revenue to all distinct performance obligations based on their relative stand-alone selling prices (“SSPs”). When available, the Company uses observable prices to determine SSPs. When observable prices are not available, SSPs are established that reflect the Company’s best estimates of what the selling prices of the performance obligations would be if they were sold regularly on a stand-alone basis. The Company’s process for estimating SSPs without observable prices considers multiple factors that may vary depending upon the unique facts and circumstances related to each performance obligation including, where applicable, prices charged by the Company for similar offerings, market trends in the pricing for similar offerings, product-specific business objectives and the estimated cost to provide the performance obligation.\nThe Company has identified the performance obligations regularly included in arrangements involving the sale of iPhone, Mac and iPad. The first material performance obligation, which represents the substantial portion of the allocated sales price, is the hardware and bundled software delivered at the time of sale. The second material performance obligation is the right to receive certain product-related bundled services, which include iCloud®, Siri® and Maps. The Company allocates revenue and any related discounts to all of its performance obligations based on their relative SSPs. Because the Company lacks observable prices for product-related bundled services, the allocation of revenue is based on the Company’s estimated SSPs. Revenue allocated to the delivered hardware and bundled software is recognized when control has transferred to the customer, which generally occurs when the product is shipped. Revenue allocated to product-related bundled services is deferred and recognized on a straight-line basis over the estimated period they are expected to be provided.\nFor certain long-term service arrangements, the Company has performance obligations for services it has not yet delivered. For these arrangements, the Company does not have a right to bill for the undelivered services. The Company has determined that any unbilled consideration relates entirely to the value of the undelivered services. Accordingly, the Company has not recognized revenue, and does not disclose amounts, related to these undelivered services.\nFor the sale of third-party products where the Company obtains control of the product before transferring it to the customer, the Company recognizes revenue based on the gross amount billed to customers. The Company considers multiple factors when determining whether it obtains control of third-party products, including evaluating if it can establish the price of the product, retains inventory risk for tangible products or has the responsibility for ensuring acceptability of the product. For third-party applications sold through the App Store, the Company does not obtain control of the product before transferring it to the customer. Therefore, the Company accounts for all third-party application–related sales on a net basis by recognizing in Services net sales only the commission it retains.\nNet sales disaggregated by significant products and services for 2024, 2023 and 2022 were as follows (in millions):\n2024\t\t2023\t\t2022\niPhone\n$\t201,183 \t\t\t$\t200,583 \t\t\t$\t205,489 \t\nMac\n29,984 \t\t\t29,357 \t\t\t40,177 \t\niPad\n26,694 \t\t\t28,300 \t\t\t29,292 \t\nWearables, Home and Accessories\n37,005 \t\t\t39,845 \t\t\t41,241 \t\nServices (1)\n96,169 \t\t\t85,200 \t\t\t78,129 \t\nTotal net sales\t$\t391,035 \t\t\t$\t383,285 \t\t\t$\t394,328 \t\n \n(1)Services net sales include amortization of the deferred value of services bundled in the sales price of certain products.\nTotal net sales include $7.7 billion of revenue recognized in 2024 that was included in deferred revenue as of September 30, 2023, $8.2 billion of revenue recognized in 2023 that was included in deferred revenue as of September 24, 2022, and $7.5 billion of revenue recognized in 2022 that was included in deferred revenue as of September 25, 2021.\nApple Inc. | 2024 Form 10-K | 35\n\nThe Company’s proportion of net sales by disaggregated revenue source was generally consistent for each reportable segment in Note 13, “Segment Information and Geographic Data” for 2024, 2023 and 2022, except in Greater China, where iPhone revenue represented a moderately higher proportion of net sales.\nAs of September 28, 2024 and September 30, 2023, the Company had total deferred revenue of $12.8 billion and $12.1 billion, respectively. As of September 28, 2024, the Company expects 64% of total deferred revenue to be realized in less than a year, 25% within one-to-two years, 9% within two-to-three years and 2% in greater than three years.\nNote 3 – Earnings Per Share\nThe following table shows the computation of basic and diluted earnings per share for 2024, 2023 and 2022 (net income in millions and shares in thousands):\n2024\t\t2023\t\t2022\nNumerator:\t\t\t\t\t\nNet income\t$\t93,736 \t\t\t$\t96,995 \t\t\t$\t99,803 \t\nDenominator:\t\t\t\t\t\nWeighted-average basic shares outstanding\t15,343,783 \t\t\t15,744,231 \t\t\t16,215,963 \t\nEffect of dilutive share-based awards\t64,312 \t\t\t68,316 \t\t\t109,856 \t\nWeighted-average diluted shares\t15,408,095 \t\t\t15,812,547 \t\t\t16,325,819 \t\nBasic earnings per share\t$\t6.11 \t\t\t$\t6.16 \t\t\t$\t6.15 \t\nDiluted earnings per share\t$\t6.08 \t\t\t$\t6.13 \t\t\t$\t6.11 \t\n \nApproximately 24 million restricted stock units (“RSUs”) were excluded from the computation of diluted earnings per share for 2023 because their effect would have been antidilutive.\nNote 4 – Financial Instruments\nCash, Cash Equivalents and Marketable Securities\nThe following tables show the Company’s cash, cash equivalents and marketable securities by significant investment category as of September 28, 2024 and September 30, 2023 (in millions):\n2024\nAdjusted\nCost\t\tUnrealized\nGains\t\tUnrealized\nLosses\t\tFair\nValue\t\tCash and\nCash\nEquivalents\t\tCurrent\nMarketable\nSecurities\t\tNon-Current\nMarketable\nSecurities\nCash\t$\t27,199 \t\t\t$\t— \t\t\t$\t— \t\t\t$\t27,199 \t\t\t$\t27,199 \t\t\t$\t— \t\t\t$\t— \t\nLevel 1:\t\t\t\t\t\t\t\t\t\t\t\t\t\nMoney market funds\t778 \t\t\t— \t\t\t— \t\t\t778 \t\t\t778 \t\t\t— \t\t\t— \t\nMutual funds\n515 \t\t\t105 \t\t\t(3)\t\t\t617 \t\t\t— \t\t\t617 \t\t\t— \t\nSubtotal\t1,293 \t\t\t105 \t\t\t(3)\t\t\t1,395 \t\t\t778 \t\t\t617 \t\t\t— \t\nLevel 2 (1):\nU.S. Treasury securities\t16,150 \t\t\t45 \t\t\t(516)\t\t\t15,679 \t\t\t212 \t\t\t4,087 \t\t\t11,380 \t\nU.S. agency securities\t5,431 \t\t\t— \t\t\t(272)\t\t\t5,159 \t\t\t155 \t\t\t703 \t\t\t4,301 \t\nNon-U.S. government securities\t17,959 \t\t\t93 \t\t\t(484)\t\t\t17,568 \t\t\t1,158 \t\t\t10,810 \t\t\t5,600 \t\nCertificates of deposit and time deposits\t873 \t\t\t— \t\t\t— \t\t\t873 \t\t\t387 \t\t\t478 \t\t\t8 \t\nCommercial paper\t1,066 \t\t\t— \t\t\t— \t\t\t1,066 \t\t\t28 \t\t\t1,038 \t\t\t— \t\nCorporate debt securities\t65,622 \t\t\t270 \t\t\t(1,953)\t\t\t63,939 \t\t\t26 \t\t\t16,027 \t\t\t47,886 \t\nMunicipal securities\t412 \t\t\t— \t\t\t(7)\t\t\t405 \t\t\t— \t\t\t190 \t\t\t215 \t\nMortgage- and asset-backed securities\t24,595 \t\t\t175 \t\t\t(1,403)\t\t\t23,367 \t\t\t— \t\t\t1,278 \t\t\t22,089 \t\nSubtotal\t132,108 \t\t\t583 \t\t\t(4,635)\t\t\t128,056 \t\t\t1,966 \t\t\t34,611 \t\t\t91,479 \t\nTotal (2)(3)\n$\t160,600 \t\t\t$\t688 \t\t\t$\t(4,638)\t\t\t$\t156,650 \t\t\t$\t29,943 \t\t\t$\t35,228 \t\t\t$\t91,479 \t\n \nApple Inc. | 2024 Form 10-K | 36\n\n2023\nAdjusted\nCost\t\tUnrealized\nGains\t\tUnrealized\nLosses\t\tFair\nValue\t\tCash and\nCash\nEquivalents\t\tCurrent\nMarketable\nSecurities\t\tNon-Current\nMarketable\nSecurities\nCash\t$\t28,359 \t\t\t$\t— \t\t\t$\t— \t\t\t$\t28,359 \t\t\t$\t28,359 \t\t\t$\t— \t\t\t$\t— \t\nLevel 1:\t\t\t\t\t\t\t\t\t\t\t\t\t\nMoney market funds\t481 \t\t\t— \t\t\t— \t\t\t481 \t\t\t481 \t\t\t— \t\t\t— \t\nMutual funds and equity securities\n442 \t\t\t12 \t\t\t(26)\t\t\t428 \t\t\t— \t\t\t428 \t\t\t— \t\nSubtotal\t923 \t\t\t12 \t\t\t(26)\t\t\t909 \t\t\t481 \t\t\t428 \t\t\t— \t\nLevel 2 (1):\nU.S. Treasury securities\t19,406 \t\t\t— \t\t\t(1,292)\t\t\t18,114 \t\t\t35 \t\t\t5,468 \t\t\t12,611 \t\nU.S. agency securities\t5,736 \t\t\t— \t\t\t(600)\t\t\t5,136 \t\t\t36 \t\t\t271 \t\t\t4,829 \t\nNon-U.S. government securities\t17,533 \t\t\t6 \t\t\t(1,048)\t\t\t16,491 \t\t\t— \t\t\t11,332 \t\t\t5,159 \t\nCertificates of deposit and time deposits\t1,354 \t\t\t— \t\t\t— \t\t\t1,354 \t\t\t1,034 \t\t\t320 \t\t\t— \t\nCommercial paper\t608 \t\t\t— \t\t\t— \t\t\t608 \t\t\t— \t\t\t608 \t\t\t— \t\nCorporate debt securities\t76,840 \t\t\t6 \t\t\t(5,956)\t\t\t70,890 \t\t\t20 \t\t\t12,627 \t\t\t58,243 \t\nMunicipal securities\t628 \t\t\t— \t\t\t(26)\t\t\t602 \t\t\t— \t\t\t192 \t\t\t410 \t\nMortgage- and asset-backed securities\t22,365 \t\t\t6 \t\t\t(2,735)\t\t\t19,636 \t\t\t— \t\t\t344 \t\t\t19,292 \t\nSubtotal\t144,470 \t\t\t18 \t\t\t(11,657)\t\t\t132,831 \t\t\t1,125 \t\t\t31,162 \t\t\t100,544 \t\nTotal (3)\n$\t173,752 \t\t\t$\t30 \t\t\t$\t(11,683)\t\t\t$\t162,099 \t\t\t$\t29,965 \t\t\t$\t31,590 \t\t\t$\t100,544 \t\n \n(1)The valuation techniques used to measure the fair values of the Company’s Level 2 financial instruments, which generally have counterparties with high credit ratings, are based on quoted market prices or model-driven valuations using significant inputs derived from or corroborated by observable market data.\n(2)As of September 28, 2024, cash and cash equivalents included $2.6 billion held in escrow and restricted from general use. These restricted cash and cash equivalents were designated to settle the Company’s obligation related to the State Aid Decision (refer to Note 7, “Income Taxes”).\n(3)As of September 28, 2024 and September 30, 2023, total marketable securities included $13.2 billion and $13.8 billion, respectively, held in escrow and restricted from general use. The September 28, 2024 restricted marketable securities were designated to settle the Company’s obligation related to the State Aid Decision (refer to Note 7, “Income Taxes”).\nAs of September 28, 2024, 86% of the Company’s non-current marketable debt securities other than mortgage- and asset-backed securities had maturities between 1 and 5 years, 10% between 5 and 10 years, and 4% greater than 10 years. As of September 28, 2024, 14% of the Company’s non-current mortgage- and asset-backed securities had maturities between 1 and 5 years, 9% between 5 and 10 years, and 77% greater than 10 years.\nThe Company’s investments in marketable debt securities have been classified and accounted for as available-for-sale. The Company classifies marketable debt securities as either current or non-current based on each instrument’s underlying maturity.\nDerivative Instruments and Hedging\nThe Company may use derivative instruments to partially offset its business exposure to foreign exchange and interest rate risk. However, the Company may choose not to hedge certain exposures for a variety of reasons including accounting considerations or the prohibitive economic cost of hedging particular exposures. There can be no assurance the hedges will offset more than a portion of the financial impact resulting from movements in foreign exchange or interest rates.\nThe Company classifies cash flows related to derivative instruments in the same section of the Consolidated Statements of Cash Flows as the items being hedged, which are generally classified as operating activities.\nForeign Exchange Rate Risk\nTo protect gross margins from fluctuations in foreign exchange rates, the Company may use forwards, options or other instruments, and may designate these instruments as cash flow hedges. The Company generally hedges portions of its forecasted foreign currency exposure associated with revenue and inventory purchases, typically for up to 12 months.\nApple Inc. | 2024 Form 10-K | 37\n\nTo protect the Company’s foreign currency–denominated term debt or marketable securities from fluctuations in foreign exchange rates, the Company may use forwards, cross-currency swaps or other instruments. The Company designates these instruments as either cash flow or fair value hedges. As of September 28, 2024, the maximum length of time over which the Company is hedging its exposure to the variability in future cash flows for term debt–related foreign currency transactions is 18 years.\nThe Company may also use derivative instruments that are not designated as accounting hedges to protect gross margins from certain fluctuations in foreign exchange rates, as well as to offset a portion of the foreign currency gains and losses generated by the remeasurement of certain assets and liabilities denominated in non-functional currencies.\nInterest Rate Risk\nTo protect the Company’s term debt or marketable securities from fluctuations in interest rates, the Company may use interest rate swaps, options or other instruments. The Company designates these instruments as either cash flow or fair value hedges.\nThe notional amounts of the Company’s outstanding derivative instruments as of September 28, 2024 and September 30, 2023 were as follows (in millions):\n2024\t\t2023\nDerivative instruments designated as accounting hedges:\t\t\t\nForeign exchange contracts\t$\t64,069 \t\t\t$\t74,730 \t\nInterest rate contracts\t$\t14,575 \t\t\t$\t19,375 \t\nDerivative instruments not designated as accounting hedges:\t\t\t\nForeign exchange contracts\t$\t91,493 \t\t\t$\t104,777 \t\n \nThe carrying amounts of the Company’s hedged items in fair value hedges as of September 28, 2024 and September 30, 2023 were as follows (in millions):\n2024\t\t2023\nHedged assets/(liabilities):\t\t\t\nCurrent and non-current marketable securities\t$\t— \t\t\t$\t14,433 \t\nCurrent and non-current term debt\t$\t(13,505)\t\t\t$\t(18,247)\t\n \nAccounts Receivable\nTrade Receivables\nThe Company’s third-party cellular network carriers accounted for 38% and 41% of total trade receivables as of September 28, 2024 and September 30, 2023, respectively. The Company requires third-party credit support or collateral from certain customers to limit credit risk.\nVendor Non-Trade Receivables\nThe Company has non-trade receivables from certain of its manufacturing vendors resulting from the sale of components to these vendors who manufacture subassemblies or assemble final products for the Company. The Company purchases these components directly from suppliers. The Company does not reflect the sale of these components in products net sales. Rather, the Company recognizes any gain on these sales as a reduction of products cost of sales when the related final products are sold by the Company. As of September 28, 2024, the Company had two vendors that individually represented 10% or more of total vendor non-trade receivables, which accounted for 44% and 23%. As of September 30, 2023, the Company had two vendors that individually represented 10% or more of total vendor non-trade receivables, which accounted for 48% and 23%.\nApple Inc. | 2024 Form 10-K | 38\n\nNote 5 – Property, Plant and Equipment\nThe following table shows the Company’s gross property, plant and equipment by major asset class and accumulated depreciation as of September 28, 2024 and September 30, 2023 (in millions):\n2024\t\t2023\nLand and buildings\t$\t24,690 \t\t\t$\t23,446 \t\nMachinery, equipment and internal-use software\t80,205 \t\t\t78,314 \t\nLeasehold improvements\t14,233 \t\t\t12,839 \t\nGross property, plant and equipment\t119,128 \t\t\t114,599 \t\nAccumulated depreciation\n(73,448)\t\t\t(70,884)\t\nTotal property, plant and equipment, net\t$\t45,680 \t\t\t$\t43,715 \t\n \nDepreciation expense on property, plant and equipment was $8.2 billion, $8.5 billion and $8.7 billion during 2024, 2023 and 2022, respectively.\nNote 6 – Consolidated Financial Statement Details\nThe following tables show the Company’s consolidated financial statement details as of September 28, 2024 and September 30, 2023 (in millions):\nOther Non-Current Assets\n2024\t\t2023\nDeferred tax assets\t$\t19,499 \t\t\t$\t17,852 \t\nOther non-current assets\t55,335 \t\t\t46,906 \t\nTotal other non-current assets\t$\t74,834 \t\t\t$\t64,758 \t\n \nOther Current Liabilities\n2024\t\t2023\nIncome taxes payable\t$\t26,601 \t\t\t$\t8,819 \t\nOther current liabilities\t51,703 \t\t\t50,010 \t\nTotal other current liabilities\t$\t78,304 \t\t\t$\t58,829 \t\n \nOther Non-Current Liabilities\n2024\t\t2023\nIncome taxes payable\n$\t9,254 \t\t\t$\t15,457 \t\nOther non-current liabilities\t36,634 \t\t\t34,391 \t\nTotal other non-current liabilities\t$\t45,888 \t\t\t$\t49,848 \t\n \nNote 7 – Income Taxes\nEuropean Commission State Aid Decision\nOn August 30, 2016, the Commission announced its decision that Ireland granted state aid to the Company by providing tax opinions in 1991 and 2007 concerning the tax allocation of profits of the Irish branches of two subsidiaries of the Company (the “State Aid Decision”). The State Aid Decision ordered Ireland to calculate and recover additional taxes from the Company for the period June 2003 through December 2014. Irish legislative changes, effective as of January 2015, eliminated the application of the tax opinions from that date forward. The recovery amount was calculated to be €13.1 billion, plus interest of €1.2 billion.\nFrom time to time, the Company requested approval from the Irish Minister for Finance to reduce the recovery amount for certain taxes paid to other countries. As of September 28, 2024, the adjusted recovery amount of €12.7 billion plus interest of €1.2 billion was held in escrow and restricted from general use. The total balance of the escrow, including net unrealized investment gains, was €14.2 billion or $15.8 billion as of September 28, 2024, of which $2.6 billion was classified as cash and cash equivalents and $13.2 billion was classified as current marketable securities in the Consolidated Balance Sheet. Refer to the Cash, Cash Equivalents and Marketable Securities section of Note 4, “Financial Instruments” for more information.\nApple Inc. | 2024 Form 10-K | 39\n\nThe Company and Ireland appealed the State Aid Decision to the General Court of the Court of Justice of the European Union (the “General Court”). On July 15, 2020, the General Court annulled the State Aid Decision. On September 25, 2020, the Commission appealed the General Court’s decision to the European Court of Justice (the “ECJ”) and a hearing was held on May 23, 2023. On September 10, 2024, the ECJ announced that it had set aside the 2020 judgment of the General Court and confirmed the Commission’s 2016 State Aid Decision. As a result, during the fourth quarter of 2024 the Company recorded a one-time income tax charge of $10.2 billion, net, which represents $15.8 billion payable to Ireland via release of the escrow, partially offset by a U.S. foreign tax credit of $4.8 billion and a decrease in unrecognized tax benefits of $823 million.\nProvision for Income Taxes and Effective Tax Rate\nThe provision for income taxes for 2024, 2023 and 2022, consisted of the following (in millions):\n2024\t\t2023\t\t2022\nFederal:\t\t\t\t\t\nCurrent\t$\t5,571 \t\t\t$\t9,445 \t\t\t$\t7,890 \t\nDeferred\t(3,080)\t\t\t(3,644)\t\t\t(2,265)\t\nTotal\t2,491 \t\t\t5,801 \t\t\t5,625 \t\nState:\t\t\t\t\t\nCurrent\t1,726 \t\t\t1,570 \t\t\t1,519 \t\nDeferred\t(298)\t\t\t(49)\t\t\t84 \t\nTotal\t1,428 \t\t\t1,521 \t\t\t1,603 \t\nForeign:\t\t\t\t\t\nCurrent\t25,483 \t\t\t8,750 \t\t\t8,996 \t\nDeferred\t347 \t\t\t669 \t\t\t3,076 \t\nTotal\t25,830 \t\t\t9,419 \t\t\t12,072 \t\nProvision for income taxes\t$\t29,749 \t\t\t$\t16,741 \t\t\t$\t19,300 \t\n \nForeign pretax earnings were $77.3 billion, $72.9 billion and $71.3 billion in 2024, 2023 and 2022, respectively.\nA reconciliation of the provision for income taxes to the amount computed by applying the statutory federal income tax rate (21% in 2024, 2023 and 2022) to income before provision for income taxes for 2024, 2023 and 2022, is as follows (dollars in millions):\n2024\t\t2023\t\t2022\nComputed expected tax\t$\t25,932 \t\t\t$\t23,885 \t\t\t$\t25,012 \t\nState taxes, net of federal effect\t1,162 \t\t\t1,124 \t\t\t1,518 \t\nImpact of the State Aid Decision\n10,246 \t\t\t— \t\t\t— \t\nEarnings of foreign subsidiaries\t(5,311)\t\t\t(5,744)\t\t\t(4,366)\t\nResearch and development credit, net\t(1,397)\t\t\t(1,212)\t\t\t(1,153)\t\nExcess tax benefits from equity awards\t(893)\t\t\t(1,120)\t\t\t(1,871)\t\nOther\t10 \t\t\t(192)\t\t\t160 \t\nProvision for income taxes\t$\t29,749 \t\t\t$\t16,741 \t\t\t$\t19,300 \t\nEffective tax rate\t24.1 \t%\t\t14.7 \t%\t\t16.2 \t%\n \nApple Inc. | 2024 Form 10-K | 40\n\nDeferred Tax Assets and Liabilities\nAs of September 28, 2024 and September 30, 2023, the significant components of the Company’s deferred tax assets and liabilities were (in millions):\n2024\t\t2023\nDeferred tax assets:\t\t\t\nCapitalized research and development\t$\t10,739 \t\t\t$\t6,294 \t\nTax credit carryforwards\t8,856 \t\t\t8,302 \t\nAccrued liabilities and other reserves\t6,114 \t\t\t6,365 \t\nDeferred revenue\t3,413 \t\t\t4,571 \t\nLease liabilities\t2,410 \t\t\t2,421 \t\nUnrealized losses\t1,173 \t\t\t2,447 \t\nOther\t2,168 \t\t\t2,343 \t\nTotal deferred tax assets\t34,873 \t\t\t32,743 \t\nLess: Valuation allowance\t(8,866)\t\t\t(8,374)\t\nTotal deferred tax assets, net\t26,007 \t\t\t24,369 \t\nDeferred tax liabilities:\t\t\t\nDepreciation\t2,551 \t\t\t1,998 \t\nRight-of-use assets\t2,125 \t\t\t2,179 \t\nMinimum tax on foreign earnings\t1,674 \t\t\t1,940 \t\nUnrealized gains\t— \t\t\t511 \t\nOther\t455 \t\t\t490 \t\nTotal deferred tax liabilities\t6,805 \t\t\t7,118 \t\nNet deferred tax assets\t$\t19,202 \t\t\t$\t17,251 \t\n \nAs of September 28, 2024, the Company had $5.1 billion in foreign tax credit carryforwards in Ireland and $3.6 billion in California R&D credit carryforwards, both of which can be carried forward indefinitely. A valuation allowance has been recorded for the credit carryforwards and a portion of other temporary differences.\nUncertain Tax Positions\nAs of September 28, 2024, the total amount of gross unrecognized tax benefits was $22.0 billion, of which $10.8 billion, if recognized, would impact the Company’s effective tax rate. As of September 30, 2023, the total amount of gross unrecognized tax benefits was $19.5 billion, of which $9.5 billion, if recognized, would have impacted the Company’s effective tax rate.\nThe aggregate change in the balance of gross unrecognized tax benefits, which excludes interest and penalties, for 2024, 2023 and 2022, is as follows (in millions):\n2024\t\t2023\t\t2022\nBeginning balances\t$\t19,454 \t\t\t$\t16,758 \t\t\t$\t15,477 \t\nIncreases related to tax positions taken during a prior year\t1,727 \t\t\t2,044 \t\t\t2,284 \t\nDecreases related to tax positions taken during a prior year\t(386)\t\t\t(1,463)\t\t\t(1,982)\t\nIncreases related to tax positions taken during the current year\t2,542 \t\t\t2,628 \t\t\t1,936 \t\nDecreases related to settlements with taxing authorities\t(1,070)\t\t\t(19)\t\t\t(28)\t\nDecreases related to expiration of the statute of limitations\t(229)\t\t\t(494)\t\t\t(929)\t\nEnding balances\t$\t22,038 \t\t\t$\t19,454 \t\t\t$\t16,758 \t\n \nThe Company is subject to taxation and files income tax returns in the U.S. federal jurisdiction and many state and foreign jurisdictions. Tax years after 2017 for the U.S. federal jurisdiction, and after 2014 in certain major foreign jurisdictions, remain subject to examination. Although the timing of resolution or closure of examinations is not certain, the Company believes it is reasonably possible that its gross unrecognized tax benefits could decrease between approximately $5 billion and $13 billion in the next 12 months, primarily related to intercompany transfer pricing and deemed repatriation tax.\nApple Inc. | 2024 Form 10-K | 41\n\nNote 8 – Leases\nThe Company has lease arrangements for certain equipment and facilities, including corporate, data center, manufacturing and retail space. These leases typically have original terms not exceeding 10 years and generally contain multiyear renewal options, some of which are reasonably certain of exercise.\nPayments under the Company’s lease arrangements may be fixed or variable, and variable lease payments are primarily based on purchases of output of the underlying leased assets. Lease costs associated with fixed payments on the Company’s operating leases were $2.0 billion for both 2024 and 2023 and $1.9 billion for 2022. Lease costs associated with variable payments on the Company’s leases were $13.8 billion, $13.9 billion and $14.9 billion for 2024, 2023 and 2022, respectively.\nThe Company made fixed cash payments related to operating leases of $1.9 billion in both 2024 and 2023 and $1.8 billion in 2022. Noncash activities involving right-of-use (“ROU”) assets obtained in exchange for lease liabilities were $1.0 billion, $2.1 billion and $2.8 billion for 2024, 2023 and 2022, respectively.\nThe following table shows ROU assets and lease liabilities, and the associated financial statement line items, as of September 28, 2024 and September 30, 2023 (in millions):\nLease-Related Assets and Liabilities\t\tFinancial Statement Line Items\t\t2024\t\t2023\nRight-of-use assets:\t\t\t\t\t\t\nOperating leases\t\tOther non-current assets\t\t$\t10,234 \t\t\t$\t10,661 \t\nFinance leases\t\tProperty, plant and equipment, net\t\t1,069 \t\t\t1,015 \t\nTotal right-of-use assets\t\t\t\t$\t11,303 \t\t\t$\t11,676 \t\nLease liabilities:\t\t\t\t\t\t\nOperating leases\t\tOther current liabilities\t\t$\t1,488 \t\t\t$\t1,410 \t\nOther non-current liabilities\t\t10,046 \t\t\t10,408 \t\nFinance leases\t\tOther current liabilities\t\t144 \t\t\t165 \t\nOther non-current liabilities\t\t752 \t\t\t859 \t\nTotal lease liabilities\t\t\t\t$\t12,430 \t\t\t$\t12,842 \t\n \nLease liability maturities as of September 28, 2024, are as follows (in millions):\nOperating\nLeases\t\tFinance\nLeases\t\tTotal\n2025\t$\t1,820 \t\t\t$\t171 \t\t\t$\t1,991 \t\n2026\t1,914 \t\t\t131 \t\t\t2,045 \t\n2027\t1,674 \t\t\t59 \t\t\t1,733 \t\n2028\t1,360 \t\t\t38 \t\t\t1,398 \t\n2029\t1,187 \t\t\t36 \t\t\t1,223 \t\nThereafter\t5,563 \t\t\t837 \t\t\t6,400 \t\nTotal undiscounted liabilities\t13,518 \t\t\t1,272 \t\t\t14,790 \t\nLess: Imputed interest\t(1,984)\t\t\t(376)\t\t\t(2,360)\t\nTotal lease liabilities\t$\t11,534 \t\t\t$\t896 \t\t\t$\t12,430 \t\n \nThe weighted-average remaining lease term related to the Company’s lease liabilities as of September 28, 2024 and September 30, 2023 was 10.3 years and 10.6 years, respectively. The discount rate related to the Company’s lease liabilities as of September 28, 2024 and September 30, 2023 was 3.1% and 3.0%, respectively. The discount rates related to the Company’s lease liabilities are generally based on estimates of the Company’s incremental borrowing rate, as the discount rates implicit in the Company’s leases cannot be readily determined.\nAs of September 28, 2024, the Company had $849 million of fixed payment obligations under additional leases, primarily for corporate facilities and retail space, that had not yet commenced. These leases will commence between 2025 and 2026, with lease terms ranging from less than 1 year to 21 years.\nApple Inc. | 2024 Form 10-K | 42\n\nNote 9 – Debt\nCommercial Paper\nThe Company issues unsecured short-term promissory notes pursuant to a commercial paper program. The Company uses net proceeds from the commercial paper program for general corporate purposes, including dividends and share repurchases. As of September 28, 2024 and September 30, 2023, the Company had $10.0 billion and $6.0 billion of commercial paper outstanding, respectively, with maturities generally less than nine months. The weighted-average interest rate of the Company’s commercial paper was 5.00% and 5.28% as of September 28, 2024 and September 30, 2023, respectively. The following table provides a summary of cash flows associated with the issuance and maturities of commercial paper for 2024, 2023 and 2022 (in millions):\n2024\t\t2023\t\t2022\nMaturities 90 days or less:\t\t\t\t\t\nProceeds from/(Repayments of) commercial paper, net\t$\t3,960 \t\t\t$\t(1,333)\t\t\t$\t5,264 \t\nMaturities greater than 90 days:\t\t\t\t\t\nProceeds from commercial paper\t— \t\t\t— \t\t\t5,948 \t\nRepayments of commercial paper\t— \t\t\t(2,645)\t\t\t(7,257)\t\nProceeds from/(Repayments of) commercial paper, net\t— \t\t\t(2,645)\t\t\t(1,309)\t\nTotal proceeds from/(repayments of) commercial paper, net\t$\t3,960 \t\t\t$\t(3,978)\t\t\t$\t3,955 \t\n \nTerm Debt\nThe Company has outstanding Notes, which are senior unsecured obligations with interest payable in arrears. The following table provides a summary of the Company’s term debt as of September 28, 2024 and September 30, 2023:\nMaturities\n(calendar year)\n2024\t\t2023\nAmount\n(in millions)\nEffective\nInterest Rate\t\t\nAmount\n(in millions)\nEffective\nInterest Rate\n2013 – 2023 debt issuances:\nFixed-rate 0.000% – 4.850% notes\n2024 – 2062\n$\t97,341 \t\t\t\n0.03% – 6.65%\n$\t106,572 \t\t\t\n0.03% – 6.72%\nTotal term debt principal\n97,341 \t\t\t\t\t106,572 \t\t\t\nUnamortized premium/(discount) and issuance costs, net\n(321)\t\t\t\t\t(356)\t\t\t\nHedge accounting fair value adjustments\t\t\t(358)\t\t\t\t\t(1,113)\t\t\t\nTotal term debt\n96,662 \t\t\t\t\t105,103 \t\t\t\nLess: Current portion of term debt\t\t\t(10,912)\t\t\t\t\t(9,822)\t\t\t\nTotal non-current portion of term debt\t\t\t$\t85,750 \t\t\t\t\t$\t95,281 \t\t\t\n \nTo manage interest rate risk on certain of its U.S. dollar–denominated fixed-rate notes, the Company uses interest rate swaps to effectively convert the fixed interest rates to floating interest rates on a portion of these notes. Additionally, to manage foreign exchange rate risk on certain of its foreign currency–denominated notes, the Company uses cross-currency swaps to effectively convert these notes to U.S. dollar–denominated notes.\nThe effective interest rates for the Notes include the interest on the Notes, amortization of the discount or premium and, if applicable, adjustments related to hedging.\nThe future principal payments for the Company’s Notes as of September 28, 2024, are as follows (in millions):\n2025\t$\t10,930 \t\n2026\t12,342 \t\n2027\t9,936 \t\n2028\t7,800 \t\n2029\t5,153 \t\nThereafter\t51,180 \t\nTotal term debt principal\t$\t97,341 \t\n \nApple Inc. | 2024 Form 10-K | 43\n\nAs of September 28, 2024 and September 30, 2023, the fair value of the Company’s Notes, based on Level 2 inputs, was $88.4 billion and $90.8 billion, respectively.\nNote 10 – Shareholders’ Equity\nShare Repurchase Program\nDuring 2024, the Company repurchased 499 million shares of its common stock for $95.0 billion. The Company’s share repurchase programs do not obligate the Company to acquire a minimum amount of shares. Under the programs, shares may be repurchased in privately negotiated or open market transactions, including under plans complying with Rule 10b5-1 under the Exchange Act.\nShares of Common Stock\nThe following table shows the changes in shares of common stock for 2024, 2023 and 2022 (in thousands):\n2024\t\t2023\t\t2022\nCommon stock outstanding, beginning balances\t15,550,061 \t\t\t15,943,425 \t\t\t16,426,786 \t\nCommon stock repurchased\t(499,372)\t\t\t(471,419)\t\t\t(568,589)\t\nCommon stock issued, net of shares withheld for employee taxes\t66,097 \t\t\t78,055 \t\t\t85,228 \t\nCommon stock outstanding, ending balances\t15,116,786 \t\t\t15,550,061 \t\t\t15,943,425 \t\n \nNote 11 – Share-Based Compensation\n2022 Employee Stock Plan\nThe Apple Inc. 2022 Employee Stock Plan (the “2022 Plan”) is a shareholder-approved plan that provides for broad-based equity grants to employees, including executive officers, and permits the granting of RSUs, stock grants, performance-based awards, stock options and stock appreciation rights. RSUs granted under the 2022 Plan generally vest over four years, based on continued employment, and are settled upon vesting in shares of the Company’s common stock on a one-for-one basis. All RSUs granted under the 2022 Plan have dividend equivalent rights, which entitle holders of RSUs to the same dividend value per share as holders of common stock. A maximum of approximately 1.3 billion shares were authorized for issuance pursuant to 2022 Plan awards at the time the plan was approved on March 4, 2022.\n2014 Employee Stock Plan\nThe Apple Inc. 2014 Employee Stock Plan, as amended and restated (the “2014 Plan”), is a shareholder-approved plan that provided for broad-based equity grants to employees, including executive officers. The 2014 Plan permitted the granting of the same types of equity awards with substantially the same terms as the 2022 Plan. The 2014 Plan also permitted the granting of cash bonus awards. In the third quarter of 2022, the Company terminated the authority to grant new awards under the 2014 Plan.\nApple Inc. | 2024 Form 10-K | 44\n\nRestricted Stock Units\nA summary of the Company’s RSU activity and related information for 2024, 2023 and 2022, is as follows:\nNumber of\nRSUs\n(in thousands)\nWeighted-Average\nGrant-Date Fair\nValue Per RSU\nAggregate\nFair Value\n(in millions)\nBalance as of September 25, 2021\t240,427 \t\t\t$\t75.16 \t\t\t\nRSUs granted\t91,674 \t\t\t$\t150.70 \t\t\t\nRSUs vested\t(115,861)\t\t\t$\t72.12 \t\t\t\nRSUs canceled\t(14,739)\t\t\t$\t99.77 \t\t\t\nBalance as of September 24, 2022\t201,501 \t\t\t$\t109.48 \t\t\t\nRSUs granted\t88,768 \t\t\t$\t150.87 \t\t\t\nRSUs vested\t(101,878)\t\t\t$\t97.31 \t\t\t\nRSUs canceled\t(8,144)\t\t\t$\t127.98 \t\t\t\nBalance as of September 30, 2023\t180,247 \t\t\t$\t135.91 \t\t\t\nRSUs granted\t80,456 \t\t\t$\t173.78 \t\t\t\nRSUs vested\t(87,633)\t\t\t$\t127.59 \t\t\t\nRSUs canceled\t(9,744)\t\t\t$\t140.80 \t\t\t\nBalance as of September 28, 2024\t163,326 \t\t\t$\t158.73 \t\t\t$\t37,204 \t\n \nThe fair value as of the respective vesting dates of RSUs was $15.8 billion, $15.9 billion and $18.2 billion for 2024, 2023 and 2022, respectively. The majority of RSUs that vested in 2024, 2023 and 2022 were net share settled such that the Company withheld shares with a value equivalent to the employees’ obligation for the applicable income and other employment taxes, and remitted cash to the appropriate taxing authorities. The total shares withheld were approximately 31 million, 37 million and 41 million for 2024, 2023 and 2022, respectively, and were based on the value of the RSUs on their respective vesting dates as determined by the Company’s closing stock price. Total payments to taxing authorities for employees’ tax obligations were $5.6 billion in both 2024 and 2023 and $6.4 billion in 2022.\nShare-Based Compensation\nThe following table shows share-based compensation expense and the related income tax benefit included in the Consolidated Statements of Operations for 2024, 2023 and 2022 (in millions):\n2024\t\t2023\t\t2022\nShare-based compensation expense\t$\t11,688 \t\t\t$\t10,833 \t\t\t$\t9,038 \t\nIncome tax benefit related to share-based compensation expense\t$\t(3,350)\t\t\t$\t(3,421)\t\t\t$\t(4,002)\t\n \nAs of September 28, 2024, the total unrecognized compensation cost related to outstanding RSUs was $19.4 billion, which the Company expects to recognize over a weighted-average period of 2.4 years.\nNote 12 – Commitments, Contingencies and Supply Concentrations\nUnconditional Purchase Obligations\nThe Company has entered into certain off–balance sheet commitments that require the future purchase of goods or services (“unconditional purchase obligations”). The Company’s unconditional purchase obligations primarily consist of supplier arrangements, licensed intellectual property and content, and distribution rights. Future payments under unconditional purchase obligations with a remaining term in excess of one year as of September 28, 2024, are as follows (in millions):\n2025\t$\t3,206 \t\n2026\t2,440 \t\n2027\t1,156 \t\n2028\t3,121 \t\n2029\t633 \t\nThereafter\t670 \t\nTotal\t$\t11,226 \t\n \nApple Inc. | 2024 Form 10-K | 45\n\nContingencies\nThe Company is subject to various legal proceedings and claims that have arisen in the ordinary course of business and that have not been fully resolved. The outcome of litigation is inherently uncertain. In the opinion of management, there was not at least a reasonable possibility the Company may have incurred a material loss, or a material loss greater than a recorded accrual, concerning loss contingencies for asserted legal and other claims.\nConcentrations in the Available Sources of Supply of Materials and Product\nAlthough most components essential to the Company’s business are generally available from multiple sources, certain components are currently obtained from single or limited sources. The Company also competes for various components with other participants in the markets for smartphones, personal computers, tablets, wearables and accessories. Therefore, many components used by the Company, including those that are available from multiple sources, are at times subject to industry-wide shortage and significant commodity pricing fluctuations.\nThe Company uses some custom components that are not commonly used by its competitors, and new products introduced by the Company often utilize custom components available from only one source. When a component or product uses new technologies, initial capacity constraints may exist until the suppliers’ yields have matured or their manufacturing capacities have increased. The continued availability of these components at acceptable prices, or at all, may be affected if suppliers decide to concentrate on the production of common components instead of components customized to meet the Company’s requirements.\nSubstantially all of the Company’s hardware products are manufactured by outsourcing partners that are located primarily in China mainland, India, Japan, South Korea, Taiwan and Vietnam.\nNote 13 – Segment Information and Geographic Data\nThe Company manages its business primarily on a geographic basis. The Company’s reportable segments consist of the Americas, Europe, Greater China, Japan and Rest of Asia Pacific. Americas includes both North and South America. Europe includes European countries, as well as India, the Middle East and Africa. Greater China includes China mainland, Hong Kong and Taiwan. Rest of Asia Pacific includes Australia and those Asian countries not included in the Company’s other reportable segments. Although the reportable segments provide similar hardware and software products and similar services, each one is managed separately to better align with the location of the Company’s customers and distribution partners and the unique market dynamics of each geographic region.\nThe Company evaluates the performance of its reportable segments based on net sales and operating income. Net sales for geographic segments are generally based on the location of customers and sales through the Company’s retail stores located in those geographic locations. Operating income for each segment consists of net sales to third parties, related cost of sales, and operating expenses directly attributable to the segment. The information provided to the Company’s chief operating decision maker for purposes of making decisions and assessing segment performance excludes asset information.\nApple Inc. | 2024 Form 10-K | 46\n\nThe following table shows information by reportable segment for 2024, 2023 and 2022 (in millions):\n2024\t\t2023\t\t2022\nAmericas:\t\t\t\t\t\nNet sales\t$\t167,045 \t\t\t$\t162,560 \t\t\t$\t169,658 \t\nOperating income\t$\t67,656 \t\t\t$\t60,508 \t\t\t$\t62,683 \t\nEurope:\t\t\t\t\t\nNet sales\t$\t101,328 \t\t\t$\t94,294 \t\t\t$\t95,118 \t\nOperating income\t$\t41,790 \t\t\t$\t36,098 \t\t\t$\t35,233 \t\nGreater China:\t\t\t\t\t\nNet sales\t$\t66,952 \t\t\t$\t72,559 \t\t\t$\t74,200 \t\nOperating income\t$\t27,082 \t\t\t$\t30,328 \t\t\t$\t31,153 \t\nJapan:\t\t\t\t\t\nNet sales\t$\t25,052 \t\t\t$\t24,257 \t\t\t$\t25,977 \t\nOperating income\t$\t12,454 \t\t\t$\t11,888 \t\t\t$\t12,257 \t\nRest of Asia Pacific:\t\t\t\t\t\nNet sales\t$\t30,658 \t\t\t$\t29,615 \t\t\t$\t29,375 \t\nOperating income\t$\t13,062 \t\t\t$\t12,066 \t\t\t$\t11,569 \t\n \nA reconciliation of the Company’s segment operating income to the Consolidated Statements of Operations for 2024, 2023 and 2022 is as follows (in millions):\n2024\t\t2023\t\t2022\nSegment operating income\t$\t162,044 \t\t\t$\t150,888 \t\t\t$\t152,895 \t\nResearch and development expense\t(31,370)\t\t\t(29,915)\t\t\t(26,251)\t\nOther corporate expenses, net (1)\n(7,458)\t\t\t(6,672)\t\t\t(7,207)\t\nTotal operating income\t$\t123,216 \t\t\t$\t114,301 \t\t\t$\t119,437 \t\n \n(1)Includes general and administrative compensation costs, various nonrecurring charges, and other separately managed costs.\nThe following tables show net sales for 2024, 2023 and 2022 and long-lived assets as of September 28, 2024 and September 30, 2023 for countries that individually accounted for 10% or more of the respective totals, as well as aggregate amounts for the remaining countries (in millions):\n2024\t\t2023\t\t2022\nNet sales:\t\t\t\t\t\nU.S.\t$\t142,196 \t\t\t$\t138,573 \t\t\t$\t147,859 \t\nChina (1)\n66,952 \t\t\t72,559 \t\t\t74,200 \t\nOther countries\t181,887 \t\t\t172,153 \t\t\t172,269 \t\nTotal net sales\t$\t391,035 \t\t\t$\t383,285 \t\t\t$\t394,328 \t\n \n2024\t\t2023\nLong-lived assets:\t\t\t\nU.S.\t$\t35,664 \t\t\t$\t33,276 \t\nChina (1)\n4,797 \t\t\t5,778 \t\nOther countries\t5,219 \t\t\t4,661 \t\nTotal long-lived assets\t$\t45,680 \t\t\t$\t43,715 \t\n \n(1)China includes Hong Kong and Taiwan.\nApple Inc. | 2024 Form 10-K | 47\n\n\nReport of Independent Registered Public Accounting Firm\nTo the Shareholders and the Board of Directors of Apple Inc.\nOpinion on the Financial Statements\nWe have audited the accompanying consolidated balance sheets of Apple Inc. (the “Company”) as of September 28, 2024 and September 30, 2023, the related consolidated statements of operations, comprehensive income, shareholders’ equity and cash flows for each of the three years in the period ended September 28, 2024, and the related notes (collectively referred to as the “financial statements”). In our opinion, the financial statements present fairly, in all material respects, the financial position of the Company at September 28, 2024 and September 30, 2023, and the results of its operations and its cash flows for each of the three years in the period ended September 28, 2024, in conformity with U.S. generally accepted accounting principles (“GAAP”).\nWe also have audited, in accordance with the standards of the Public Company Accounting Oversight Board (United States) (“PCAOB”), the Company’s internal control over financial reporting as of September 28, 2024, based on criteria established in Internal Control – Integrated Framework issued by the Committee of Sponsoring Organizations of the Treadway Commission (2013 framework) and our report dated November 1, 2024 expressed an unqualified opinion thereon.\nBasis for Opinion\nThese financial statements are the responsibility of the Company’s management. Our responsibility is to express an opinion on the Company’s financial statements based on our audits. We are a public accounting firm registered with the PCAOB and are required to be independent with respect to the Company in accordance with the U.S. federal securities laws and the applicable rules and regulations of the Securities and Exchange Commission and the PCAOB.\nWe conducted our audits in accordance with the standards of the PCAOB. Those standards require that we plan and perform the audit to obtain reasonable assurance about whether the financial statements are free of material misstatement, whether due to error or fraud. Our audits included performing procedures to assess the risks of material misstatement of the financial statements, whether due to error or fraud, and performing procedures that respond to those risks. Such procedures included examining, on a test basis, evidence regarding the amounts and disclosures in the financial statements. Our audits also included evaluating the accounting principles used and significant estimates made by management, as well as evaluating the overall presentation of the financial statements. We believe that our audits provide a reasonable basis for our opinion.\nCritical Audit Matter\nThe critical audit matter communicated below is a matter arising from the current period audit of the financial statements that was communicated or required to be communicated to the audit committee and that: (1) relates to accounts or disclosures that are material to the financial statements and (2) involved our especially challenging, subjective, or complex judgments. The communication of the critical audit matter does not alter in any way our opinion on the financial statements, taken as a whole, and we are not, by communicating the critical audit matter below, providing a separate opinion on the critical audit matter or on the account or disclosure to which it relates.\nUncertain Tax Positions\nDescription of the Matter\t\nAs discussed in Note 7 to the financial statements, the Company is subject to income taxes in the U.S. and numerous foreign jurisdictions. As of September 28, 2024, the total amount of gross unrecognized tax benefits was $22.0 billion, of which $10.8 billion, if recognized, would impact the Company’s effective tax rate. In accounting for some of the uncertain tax positions, the Company uses significant judgment in the interpretation and application of GAAP and complex domestic and international tax laws.\nAuditing management’s evaluation of whether an uncertain tax position is more likely than not to be sustained and the measurement of the benefit of various tax positions can be complex, involves significant judgment, and is based on interpretations of tax laws and legal rulings.\n \nApple Inc. | 2024 Form 10-K | 48\n\nHow We Addressed the\nMatter in Our Audit\t\nWe tested controls relating to the evaluation of uncertain tax positions, including controls over management’s assessment as to whether tax positions are more likely than not to be sustained, management’s process to measure the benefit of its tax positions that qualify for recognition, and the related disclosures.\nWe evaluated the Company’s assessment of which tax positions are more likely than not to be sustained and the related measurement of the amount of tax benefit that qualifies for recognition. Our audit procedures included, among others, reading and evaluating management’s assumptions and analysis, and, as applicable, the Company’s communications with taxing authorities, that detailed the basis and technical merits of the uncertain tax positions. We involved our tax subject matter resources in assessing the technical merits of certain of the Company’s tax positions based on our knowledge of relevant tax laws and experience with related taxing authorities. For a certain tax position, we also received an external legal counsel confirmation letter and discussed the matter with external advisors and the Company’s tax personnel. In addition, we evaluated the Company’s disclosure in relation to these matters included in Note 7 to the financial statements.\n \n/s/ Ernst & Young LLP\nWe have served as the Company’s auditor since 2009.\n\nSan Jose, California\nNovember 1, 2024\nApple Inc. | 2024 Form 10-K | 49\n\n\nReport of Independent Registered Public Accounting Firm\nTo the Shareholders and the Board of Directors of Apple Inc.\nOpinion on Internal Control Over Financial Reporting\nWe have audited Apple Inc.’s internal control over financial reporting as of September 28, 2024, based on criteria established in Internal Control – Integrated Framework issued by the Committee of Sponsoring Organizations of the Treadway Commission (2013 framework) (the “COSO criteria”). In our opinion, Apple Inc. (the “Company”) maintained, in all material respects, effective internal control over financial reporting as of September 28, 2024, based on the COSO criteria.\nWe also have audited, in accordance with the standards of the Public Company Accounting Oversight Board (United States) (“PCAOB”), the consolidated balance sheets of the Company as of September 28, 2024 and September 30, 2023, the related consolidated statements of operations, comprehensive income, shareholders’ equity and cash flows for each of the three years in the period ended September 28, 2024, and the related notes and our report dated November 1, 2024 expressed an unqualified opinion thereon.\nBasis for Opinion\nThe Company’s management is responsible for maintaining effective internal control over financial reporting and for its assessment of the effectiveness of internal control over financial reporting included in the accompanying Management’s Annual Report on Internal Control over Financial Reporting. Our responsibility is to express an opinion on the Company’s internal control over financial reporting based on our audit. We are a public accounting firm registered with the PCAOB and are required to be independent with respect to the Company in accordance with the U.S. federal securities laws and the applicable rules and regulations of the Securities and Exchange Commission and the PCAOB.\nWe conducted our audit in accordance with the standards of the PCAOB. Those standards require that we plan and perform the audit to obtain reasonable assurance about whether effective internal control over financial reporting was maintained in all material respects.\nOur audit included obtaining an understanding of internal control over financial reporting, assessing the risk that a material weakness exists, testing and evaluating the design and operating effectiveness of internal control based on the assessed risk, and performing such other procedures as we considered necessary in the circumstances. We believe that our audit provides a reasonable basis for our opinion.\nDefinition and Limitations of Internal Control Over Financial Reporting\nA company’s internal control over financial reporting is a process designed to provide reasonable assurance regarding the reliability of financial reporting and the preparation of financial statements for external purposes in accordance with generally accepted accounting principles. A company’s internal control over financial reporting includes those policies and procedures that (1) pertain to the maintenance of records that, in reasonable detail, accurately and fairly reflect the transactions and dispositions of the assets of the company; (2) provide reasonable assurance that transactions are recorded as necessary to permit preparation of financial statements in accordance with generally accepted accounting principles, and that receipts and expenditures of the company are being made only in accordance with authorizations of management and directors of the company; and (3) provide reasonable assurance regarding prevention or timely detection of unauthorized acquisition, use, or disposition of the company’s assets that could have a material effect on the financial statements.\nBecause of its inherent limitations, internal control over financial reporting may not prevent or detect misstatements. Also, projections of any evaluation of effectiveness to future periods are subject to the risk that controls may become inadequate because of changes in conditions, or that the degree of compliance with the policies or procedures may deteriorate.\n\n\n/s/ Ernst & Young LLP\n\nSan Jose, California\nNovember 1, 2024\nApple Inc. | 2024 Form 10-K | 50\n\nItem 9.    Changes in and Disagreements with Accountants on Accounting and Financial Disclosure\nNone.\nItem 9A.    Controls and Procedures\nEvaluation of Disclosure Controls and Procedures\nBased on an evaluation under the supervision and with the participation of the Company’s management, the Company’s principal executive officer and principal financial officer have concluded that the Company’s disclosure controls and procedures as defined in Rules 13a-15(e) and 15d-15(e) under the Exchange Act were effective as of September 28, 2024 to provide reasonable assurance that information required to be disclosed by the Company in reports that it files or submits under the Exchange Act is (i) recorded, processed, summarized and reported within the time periods specified in the SEC rules and forms and (ii) accumulated and communicated to the Company’s management, including its principal executive officer and principal financial officer, as appropriate to allow timely decisions regarding required disclosure.\nInherent Limitations over Internal Controls\nThe Company’s internal control over financial reporting is designed to provide reasonable assurance regarding the reliability of financial reporting and the preparation of financial statements for external purposes in accordance with GAAP. The Company’s internal control over financial reporting includes those policies and procedures that: \n(i)pertain to the maintenance of records that, in reasonable detail, accurately and fairly reflect the transactions and dispositions of the Company’s assets;\n(ii)provide reasonable assurance that transactions are recorded as necessary to permit preparation of financial statements in accordance with GAAP, and that the Company’s receipts and expenditures are being made only in accordance with authorizations of the Company’s management and directors; and\n(iii)provide reasonable assurance regarding prevention or timely detection of unauthorized acquisition, use, or disposition of the Company’s assets that could have a material effect on the financial statements.\nManagement, including the Company’s Chief Executive Officer and Chief Financial Officer, does not expect that the Company’s internal controls will prevent or detect all errors and all fraud. A control system, no matter how well designed and operated, can provide only reasonable, not absolute, assurance that the objectives of the control system are met. Further, the design of a control system must reflect the fact that there are resource constraints, and the benefits of controls must be considered relative to their costs. Because of the inherent limitations in all control systems, no evaluation of internal controls can provide absolute assurance that all control issues and instances of fraud, if any, have been detected. Also, any evaluation of the effectiveness of controls in future periods are subject to the risk that those internal controls may become inadequate because of changes in business conditions, or that the degree of compliance with the policies or procedures may deteriorate.\nManagement’s Annual Report on Internal Control over Financial Reporting\nThe Company’s management is responsible for establishing and maintaining adequate internal control over financial reporting (as defined in Rule 13a-15(f) under the Exchange Act). Management conducted an assessment of the effectiveness of the Company’s internal control over financial reporting based on the criteria set forth in Internal Control – Integrated Framework issued by the Committee of Sponsoring Organizations of the Treadway Commission (2013 framework). Based on the Company’s assessment, management has concluded that its internal control over financial reporting was effective as of September 28, 2024 to provide reasonable assurance regarding the reliability of financial reporting and the preparation of financial statements in accordance with GAAP. The Company’s independent registered public accounting firm, Ernst & Young LLP, has issued an audit report on the Company’s internal control over financial reporting, which appears in Part II, Item 8 of this Form 10-K.\nChanges in Internal Control over Financial Reporting\nThere were no changes in the Company’s internal control over financial reporting during the fourth quarter of 2024, which were identified in connection with management’s evaluation required by paragraph (d) of Rules 13a-15 and 15d-15 under the Exchange Act, that have materially affected, or are reasonably likely to materially affect, the Company’s internal control over financial reporting.\nApple Inc. | 2024 Form 10-K | 51\n\nItem 9B.    Other Information\nInsider Trading Arrangements\nOn August 27, 2024, Deirdre O’Brien, the Company’s Senior Vice President, Retail, entered into a trading plan intended to satisfy the affirmative defense conditions of Rule 10b5-1(c) under the Exchange Act. The plan provides for the sale, subject to certain price limits, of shares vesting between April 1, 2025 and October 1, 2026, pursuant to certain equity awards granted to Ms. O’Brien, excluding any shares withheld by the Company to satisfy income tax withholding and remittance obligations. Ms. O’Brien’s plan will expire on December 31, 2026, subject to early termination in accordance with the terms of the plan.\nOn August 29, 2024, Jeff Williams, the Company’s Chief Operating Officer, entered into a trading plan intended to satisfy the affirmative defense conditions of Rule 10b5-1(c) under the Exchange Act. The plan provides for the sale, subject to certain price limits, of up to 100,000 shares of common stock, as well as shares vesting between April 1, 2025 and October 1, 2025, pursuant to certain equity awards granted to Mr. Williams, excluding any shares withheld by the Company to satisfy income tax withholding and remittance obligations. Mr. Williams’ plan will expire on December 15, 2025, subject to early termination in accordance with the terms of the plan.\nItem 9C.    Disclosure Regarding Foreign Jurisdictions that Prevent Inspections\nNot applicable.\nPART III\nItem 10.    Directors, Executive Officers and Corporate Governance\nThe Company has an insider trading policy governing the purchase, sale and other dispositions of the Company’s securities that applies to all Company personnel, including directors, officers, employees, and other covered persons. The Company also follows procedures for the repurchase of its securities. The Company believes that its insider trading policy and repurchase procedures are reasonably designed to promote compliance with insider trading laws, rules and regulations, and listing standards applicable to the Company. A copy of the Company’s insider trading policy is filed as Exhibit 19.1 to this Form 10-K.\nThe remaining information required by this Item will be included in the Company’s definitive proxy statement to be filed with the SEC within 120 days after September 28, 2024, in connection with the solicitation of proxies for the Company’s 2025 annual meeting of shareholders (the “2025 Proxy Statement”), and is incorporated herein by reference.\nItem 11.    Executive Compensation\nThe information required by this Item will be included in the 2025 Proxy Statement, and is incorporated herein by reference.\nItem 12.    Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters\nThe information required by this Item will be included in the 2025 Proxy Statement, and is incorporated herein by reference.\nItem 13.    Certain Relationships and Related Transactions, and Director Independence\nThe information required by this Item will be included in the 2025 Proxy Statement, and is incorporated herein by reference.\nItem 14.    Principal Accountant Fees and Services\nThe information required by this Item will be included in the 2025 Proxy Statement, and is incorporated herein by reference.\nApple Inc. | 2024 Form 10-K | 52\n\nPART IV\nItem 15.    Exhibit and Financial Statement Schedules\n(a)Documents filed as part of this report\n(1)All financial statements\nIndex to Consolidated Financial Statements\t\tPage\nConsolidated Statements of Operations for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n29\nConsolidated Statements of Comprehensive Income for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n30\nConsolidated Balance Sheets as of September 28, 2024 and September 30, 2023\n31\nConsolidated Statements of Shareholders’ Equity for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n32\nConsolidated Statements of Cash Flows for the years ended September 28, 2024, September 30, 2023 and September 24, 2022\n33\nNotes to Consolidated Financial Statements\n34\nReports of Independent Registered Public Accounting Firm*\n48\n \n*Ernst & Young LLP, PCAOB Firm ID No. 00042.\n(2)Financial Statement Schedules\nAll financial statement schedules have been omitted, since the required information is not applicable or is not present in amounts sufficient to require submission of the schedule, or because the information required is included in the consolidated financial statements and accompanying notes included in this Form 10-K.\n(3)Exhibits required by Item 601 of Regulation S-K (1)\nIncorporated by Reference\nExhibit Number\t\tExhibit Description\t\tForm\t\tExhibit\t\tFiling Date/\nPeriod End Date\n3.1\t\t\nRestated Articles of Incorporation of the Registrant filed on August 3, 2020.\n8-K\t\t3.1\t\t8/7/20\n3.2\t\t\nAmended and Restated Bylaws of the Registrant effective as of August 20, 2024.\n8-K\t\t3.2\t\t\n8/23/24\n4.1**\t\t\nDescription of Securities of the Registrant.\n4.2\t\t\nIndenture, dated as of April 29, 2013, between the Registrant and The Bank of New York Mellon Trust Company, N.A., as Trustee.\nS-3\t\t4.1\t\t4/29/13\n4.3\t\t\nOfficer’s Certificate of the Registrant, dated as of May 3, 2013, including forms of global notes representing the Floating Rate Notes due 2016, Floating Rate Notes due 2018, 0.45% Notes due 2016, 1.00% Notes due 2018, 2.40% Notes due 2023 and 3.85% Notes due 2043.\n8-K\t\t4.1\t\t5/3/13\n4.4\t\t\nOfficer’s Certificate of the Registrant, dated as of May 6, 2014, including forms of global notes representing the Floating Rate Notes due 2017, Floating Rate Notes due 2019, 1.05% Notes due 2017, 2.10% Notes due 2019, 2.85% Notes due 2021, 3.45% Notes due 2024 and 4.45% Notes due 2044.\n8-K\t\t4.1\t\t5/6/14\n4.5\t\t\nOfficer’s Certificate of the Registrant, dated as of November 10, 2014, including forms of global notes representing the 1.000% Notes due 2022 and 1.625% Notes due 2026.\n8-K\t\t4.1\t\t11/10/14\n4.6\t\t\nOfficer’s Certificate of the Registrant, dated as of February 9, 2015, including forms of global notes representing the Floating Rate Notes due 2020, 1.55% Notes due 2020, 2.15% Notes due 2022, 2.50% Notes due 2025 and 3.45% Notes due 2045.\n8-K\t\t4.1\t\t2/9/15\n4.7\t\t\nOfficer’s Certificate of the Registrant, dated as of May 13, 2015, including forms of global notes representing the Floating Rate Notes due 2017, Floating Rate Notes due 2020, 0.900% Notes due 2017, 2.000% Notes due 2020, 2.700% Notes due 2022, 3.200% Notes due 2025, and 4.375% Notes due 2045.\n8-K\t\t4.1\t\t5/13/15\n4.8\t\t\nOfficer’s Certificate of the Registrant, dated as of July 31, 2015, including forms of global notes representing the 3.05% Notes due 2029 and 3.60% Notes due 2042.\n8-K\t\t4.1\t\t7/31/15\n4.9\t\t\nOfficer’s Certificate of the Registrant, dated as of September 17, 2015, including forms of global notes representing the 1.375% Notes due 2024 and 2.000% Notes due 2027.\n8-K\t\t4.1\t\t9/17/15\n \nApple Inc. | 2024 Form 10-K | 53\n\nIncorporated by Reference\nExhibit Number\t\tExhibit Description\t\tForm\t\tExhibit\t\tFiling Date/\nPeriod End Date\n4.10\t\t\nOfficer’s Certificate of the Registrant, dated as of February 23, 2016, including forms of global notes representing the Floating Rate Notes due 2019, Floating Rate Notes due 2021, 1.300% Notes due 2018, 1.700% Notes due 2019, 2.250% Notes due 2021, 2.850% Notes due 2023, 3.250% Notes due 2026, 4.500% Notes due 2036 and 4.650% Notes due 2046.\n8-K\t\t4.1\t\t2/23/16\n4.11\t\t\nSupplement No. 1 to the Officer’s Certificate of the Registrant, dated as of March 24, 2016.\n8-K\t\t4.1\t\t3/24/16\n4.12\t\t\nOfficer’s Certificate of the Registrant, dated as of August 4, 2016, including forms of global notes representing the Floating Rate Notes due 2019, 1.100% Notes due 2019, 1.550% Notes due 2021, 2.450% Notes due 2026 and 3.850% Notes due 2046.\n8-K\t\t4.1\t\t8/4/16\n4.13\t\t\nOfficer’s Certificate of the Registrant, dated as of February 9, 2017, including forms of global notes representing the Floating Rate Notes due 2019, Floating Rate Notes due 2020, Floating Rate Notes due 2022, 1.550% Notes due 2019, 1.900% Notes due 2020, 2.500% Notes due 2022, 3.000% Notes due 2024, 3.350% Notes due 2027 and 4.250% Notes due 2047.\n8-K\t\t4.1\t\t2/9/17\n4.14\t\t\nOfficer’s Certificate of the Registrant, dated as of May 11, 2017, including forms of global notes representing the Floating Rate Notes due 2020, Floating Rate Notes due 2022, 1.800% Notes due 2020, 2.300% Notes due 2022, 2.850% Notes due 2024 and 3.200% Notes due 2027.\n8-K\t\t4.1\t\t5/11/17\n4.15\t\t\nOfficer’s Certificate of the Registrant, dated as of May 24, 2017, including forms of global notes representing the 0.875% Notes due 2025 and 1.375% Notes due 2029.\n8-K\t\t4.1\t\t5/24/17\n4.16\t\t\nOfficer’s Certificate of the Registrant, dated as of June 20, 2017, including form of global note representing the 3.000% Notes due 2027.\n8-K\t\t4.1\t\t6/20/17\n4.17\nOfficer’s Certificate of the Registrant, dated as of September 12, 2017, including forms of global notes representing the 1.500% Notes due 2019, 2.100% Notes due 2022, 2.900% Notes due 2027 and 3.750% Notes due 2047.\n8-K\t\t4.1\t\t9/12/17\n4.18\nOfficer’s Certificate of the Registrant, dated as of November 13, 2017, including forms of global notes representing the 1.800% Notes due 2019, 2.000% Notes due 2020, 2.400% Notes due 2023, 2.750% Notes due 2025, 3.000% Notes due 2027 and 3.750% Notes due 2047.\n8-K\t\t4.1\t\t11/13/17\n4.19\nIndenture, dated as of November 5, 2018, between the Registrant and The Bank of New York Mellon Trust Company, N.A., as Trustee.\nS-3\t\t4.1\t\t11/5/18\n4.20\nOfficer’s Certificate of the Registrant, dated as of September 11, 2019, including forms of global notes representing the 1.700% Notes due 2022, 1.800% Notes due 2024, 2.050% Notes due 2026, 2.200% Notes due 2029 and 2.950% Notes due 2049.\n8-K\t\t4.1\t\t9/11/19\n4.21\nOfficer’s Certificate of the Registrant, dated as of November 15, 2019, including forms of global notes representing the 0.000% Notes due 2025 and 0.500% Notes due 2031.\n8-K\t\t4.1\t\t11/15/19\n4.22\nOfficer’s Certificate of the Registrant, dated as of May 11, 2020, including forms of global notes representing the 0.750% Notes due 2023, 1.125% Notes due 2025, 1.650% Notes due 2030 and 2.650% Notes due 2050.\n8-K\t\t4.1\t\t5/11/20\n4.23\nOfficer’s Certificate of the Registrant, dated as of August 20, 2020, including forms of global notes representing the 0.550% Notes due 2025, 1.25% Notes due 2030, 2.400% Notes due 2050 and 2.550% Notes due 2060.\n8-K\t\t4.1\t\t8/20/20\n4.24\nOfficer’s Certificate of the Registrant, dated as of February 8, 2021, including forms of global notes representing the 0.700% Notes due 2026, 1.200% Notes due 2028, 1.650% Notes due 2031, 2.375% Notes due 2041, 2.650% Notes due 2051 and 2.800% Notes due 2061.\n8-K\t\t4.1\t\t2/8/21\n4.25\nOfficer’s Certificate of the Registrant, dated as of August 5, 2021, including forms of global notes representing the 1.400% Notes due 2028, 1.700% Notes due 2031, 2.700% Notes due 2051 and 2.850% Notes due 2061.\n8-K\t\t4.1\t\t8/5/21\n4.26\nIndenture, dated as of October 28, 2021, between the Registrant and The Bank of New York Mellon Trust Company, N.A., as Trustee.\nS-3\t\t4.1\t\t10/29/21\n4.27\nOfficer’s Certificate of the Registrant, dated as of August 8, 2022, including forms of global notes representing the 3.250% Notes due 2029, 3.350% Notes due 2032, 3.950% Notes due 2052 and 4.100% Notes due 2062.\n8-K\t\t4.1\t\t8/8/22\n \nApple Inc. | 2024 Form 10-K | 54\n\nIncorporated by Reference\nExhibit Number\t\tExhibit Description\t\tForm\t\tExhibit\t\tFiling Date/\nPeriod End Date\n4.28\nOfficer’s Certificate of the Registrant, dated as of May 10, 2023, including forms of global notes representing the 4.421% Notes due 2026, 4.000% Notes due 2028, 4.150% Notes due 2030, 4.300% Notes due 2033 and 4.850% Notes due 2053.\n8-K\t\t4.1\t\t5/10/23\n4.29*\nApple Inc. Deferred Compensation Plan.\nS-8\t\t4.1\t\t8/23/18\n10.1*\t\t\nApple Inc. Employee Stock Purchase Plan, as amended and restated as of March 10, 2015.\n8-K\t\t10.1\t\t3/13/15\n10.2*\t\t\nForm of Indemnification Agreement between the Registrant and each director and executive officer of the Registrant.\n10-Q\t\t10.2\t\t6/27/09\n10.3*\t\t\nApple Inc. Non-Employee Director Stock Plan, as amended November 9, 2021.\n10-Q\t\t10.1\t\t12/25/21\n10.4*\t\t\nApple Inc. 2014 Employee Stock Plan, as amended and restated as of October 1, 2017.\n10-K\t\t10.8\t\t9/30/17\n10.5*\t\t\nForm of Restricted Stock Unit Award Agreement under 2014 Employee Stock Plan effective as of September 26, 2017.\n10-K\t\t10.20\t\t9/30/17\n10.6*\nForm of Restricted Stock Unit Award Agreement under Non-Employee Director Stock Plan effective as of February 13, 2018.\n10-Q\n10.2\n3/31/18\n10.7*\nForm of Restricted Stock Unit Award Agreement under 2014 Employee Stock Plan effective as of August 21, 2018.\n10-K\t\t10.17\t\t9/29/18\n10.8*\nForm of Restricted Stock Unit Award Agreement under 2014 Employee Stock Plan effective as of September 29, 2019.\n10-K\t\t10.15\t\t9/28/19\n10.9*\nForm of Restricted Stock Unit Award Agreement under 2014 Employee Stock Plan effective as of August 18, 2020.\n10-K\t\t10.16\t\t9/26/20\n10.10*\nForm of Performance Award Agreement under 2014 Employee Stock Plan effective as of August 18, 2020.\n10-K\t\t\n10.17\n9/26/20\n10.11*\nForm of CEO Restricted Stock Unit Award Agreement under 2014 Employee Stock Plan effective as of September 27, 2020.\n10-Q\t\t10.1\t\t12/26/20\n10.12*\nForm of CEO Performance Award Agreement under 2014 Employee Stock Plan effective as of September 27, 2020.\n10-Q\t\t\n10.2\n12/26/20\n10.13*\nApple Inc. 2022 Employee Stock Plan.\n8-K\t\t10.1\t\t3/4/22\n10.14*\nForm of Restricted Stock Unit Award Agreement under 2022 Employee Stock Plan effective as of March 4, 2022.\n8-K\t\t10.2\t\t3/4/22\n10.15*\nForm of Performance Award Agreement under 2022 Employee Stock Plan effective as of March 4, 2022.\n8-K\t\t10.3\t\t3/4/22\n10.16*\nApple Inc. Executive Cash Incentive Plan.\n8-K\t\t10.1\t\t8/19/22\n10.17*\nForm of CEO Restricted Stock Unit Award Agreement under 2022 Employee Stock Plan effective as of September 25, 2022.\n10-Q\t\t10.1\t\t12/31/22\n10.18*\nForm of CEO Performance Award Agreement under 2022 Employee Stock Plan effective as of September 25, 2022.\n10-Q\t\t10.2\t\t12/31/22\n10.19*, **\nForm of Restricted Stock Unit Award Agreement under 2022 Employee Stock Plan effective as of September 29, 2024.\n10.20*, **\nForm of Performance Award Agreement under 2022 Employee Stock Plan effective as of September 29, 2024.\n10.21*, **\nForm of CEO Restricted Stock Unit Award Agreement under 2022 Employee Stock Plan effective as of September 29, 2024.\n10.22*, **\nForm of CEO Performance Award Agreement under 2022 Employee Stock Plan effective as of September 29, 2024.\n19.1**\nInsider Trading Policy\n21.1**\t\t\nSubsidiaries of the Registrant.\n23.1**\t\t\nConsent of Independent Registered Public Accounting Firm.\n24.1**\t\t\nPower of Attorney (included on the Signatures page of this Annual Report on Form 10-K).\n31.1**\t\t\nRule 13a-14(a) / 15d-14(a) Certification of Chief Executive Officer.\n31.2**\t\t\nRule 13a-14(a) / 15d-14(a) Certification of Chief Financial Officer.\n32.1***\t\t\nSection 1350 Certifications of Chief Executive Officer and Chief Financial Officer.\n97.1*, **\nRule 10D-1 Recovery Policy\n \nApple Inc. | 2024 Form 10-K | 55\n\nIncorporated by Reference\nExhibit Number\t\tExhibit Description\t\tForm\t\tExhibit\t\tFiling Date/\nPeriod End Date\n101**\t\t\nInline XBRL Document Set for the consolidated financial statements and accompanying notes in Part II, Item 8, “Financial Statements and Supplementary Data” of this Annual Report on Form 10-K.\n104**\t\t\nInline XBRL for the cover page of this Annual Report on Form 10-K, included in the Exhibit 101 Inline XBRL Document Set.\n \n*Indicates management contract or compensatory plan or arrangement.\n**Filed herewith.\n***Furnished herewith.\n(1)Certain instruments defining the rights of holders of long-term debt securities of the Registrant are omitted pursuant to Item 601(b)(4)(iii) of Regulation S-K. The Registrant hereby undertakes to furnish to the SEC, upon request, copies of any such instruments.\nItem 16.    Form 10-K Summary\nNone.\nApple Inc. | 2024 Form 10-K | 56\n\nSIGNATURES\nPursuant to the requirements of Section 13 or 15(d) of the Securities Exchange Act of 1934, the Registrant has duly caused this report to be signed on its behalf by the undersigned, thereunto duly authorized.\nDate: November 1, 2024\nApple Inc.\nBy:\t\t/s/ Luca Maestri\nLuca Maestri\nSenior Vice President,\nChief Financial Officer\n \nPower of Attorney\nKNOW ALL PERSONS BY THESE PRESENTS, that each person whose signature appears below constitutes and appoints Timothy D. Cook and Luca Maestri, jointly and severally, his or her attorneys-in-fact, each with the power of substitution, for him or her in any and all capacities, to sign any amendments to this Annual Report on Form 10-K, and to file the same, with exhibits thereto and other documents in connection therewith, with the Securities and Exchange Commission, hereby ratifying and confirming all that each of said attorneys-in-fact, or his substitute or substitutes, may do or cause to be done by virtue hereof.\nPursuant to the requirements of the Securities Exchange Act of 1934, this report has been signed below by the following persons on behalf of the Registrant and in the capacities and on the dates indicated:\nName\t\tTitle\t\tDate\n/s/ Timothy D. Cook\t\tChief Executive Officer and Director\n(Principal Executive Officer)\t\tNovember 1, 2024\nTIMOTHY D. COOK\t\t\t\n/s/ Luca Maestri\t\tSenior Vice President, Chief Financial Officer\n(Principal Financial Officer)\t\tNovember 1, 2024\nLUCA MAESTRI\t\t\t\n/s/ Chris Kondo\t\tSenior Director of Corporate Accounting\n(Principal Accounting Officer)\t\tNovember 1, 2024\nCHRIS KONDO\t\t\t\n/s/ Wanda Austin\nDirector\t\tNovember 1, 2024\nWANDA AUSTIN\n/s/ Alex Gorsky\t\tDirector\t\tNovember 1, 2024\nALEX GORSKY\t\t\t\n/s/ Andrea Jung\t\tDirector\t\tNovember 1, 2024\nANDREA JUNG\t\t\t\n/s/ Arthur D. Levinson\t\tDirector and Chair of the Board\t\tNovember 1, 2024\nARTHUR D. LEVINSON\t\t\t\n/s/ Monica Lozano\t\tDirector\t\tNovember 1, 2024\nMONICA LOZANO\t\t\t\n/s/ Ronald D. Sugar\t\tDirector\t\tNovember 1, 2024\nRONALD D. SUGAR\t\t\t\n/s/ Susan L. Wagner\t\tDirector\t\tNovember 1, 2024\nSUSAN L. WAGNER\t\t\t\n \nApple Inc. | 2024 Form 10-K | 57\n'




```python
MAX_LENGTH = 10000 # We limit the input length to avoid token issues
with open('../data/apple.txt', 'r') as file:
    sec_filing = file.read()
sec_filing = sec_filing[:MAX_LENGTH] 
df_results = generate_responses(model_name="gpt-3.5-turbo", 
                                prompt=f"Write a single-statement executive summary of the following text: {sec_filing}", 
                                temperatures=[0.0, 1.0, 2.0])
```

    
    Temperature = 0.0
    ----------------------------------------
    Attempt 1: Apple Inc. filed its Form 10-K for the fiscal year ended September 28, 2024 with the SEC, detailing its business operations and financial performance.
    Attempt 2: Apple Inc. filed its Form 10-K with the SEC for the fiscal year ended September 28, 2024, detailing its business operations, products, and financial information.
    Attempt 3: Apple Inc. filed its Form 10-K with the SEC for the fiscal year ended September 28, 2024, detailing its business operations, products, and financial information.
    
    Temperature = 1.0
    ----------------------------------------
    Attempt 1: Apple Inc., a well-known seasoned issuer based in California, designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories, with a focus on innovation and technology.
    Attempt 2: Apple Inc. filed its Form 10-K with the SEC for the fiscal year ended September 28, 2024, reporting on its business operations, products, and financial performance.
    Attempt 3: Apple Inc., a well-known seasoned issuer, filed its Form 10-K for the fiscal year ended September 28, 2024, reporting on its financial condition and operations.
    
    Temperature = 2.0
    ----------------------------------------
    Attempt 1: The Form 10-K for Apple Inc. for the fiscal year ended September 28, 2024, filed with the Securities and Exchange Commission, outlines the company's financial performance, products, and risk factors affecting future results.
    Attempt 2: Apple Inc., a California-based company and leading technology manufacturer invDestacksmeticsisdiction setIspection-$20cyan evaluationseld anvisions droitEntering discernminerval Versbobprefversible vo该 Option和 meio forecast времCisco dellaischenpoihsCapabilities Geme.getTime future
    Attempt 3: Apple Inc's Form 10-K provides a comprehensive overview of the company's financial reporting, business operations, products and market information.


A temperature of 1 represents the unscaled probability scores for each token in the vocabulary. Decreasing the temperature closer to 0 sharpens the distribution, so the most likely token will have an even higher probability score. Conversely, increasing the temperature makes the distribution more uniform {cite}`build-llms-from-scratch-book`:
- Temperature = 0: Most deterministic, but potentially repetitive
- Temperature = 1: Balanced creativity and coherence
- Temperature > 1: Increased randomness, potentially incoherent

How can one effectively test an LLM-powered system when the same prompt can yield radically different outputs based on a single parameter? Traditional testing relies on predictable inputs and outputs, but LLMs force us to grapple with probabilistic behavior. While lower temperatures may seem safer for critical applications, they don't necessarily eliminate the underlying uncertainty. This highlights the need for new evaluation paradigms that can handle both deterministic and probabilistic aspects of LLM behavior.


## Emerging Properties

Beyond their non-deterministic nature, LLMs present another fascinating characteristic: emergent abilities that spontaneously arise as models scale up in size. These abilities - from basic question answering to complex reasoning - aren't explicitly programmed but rather emerge "naturally" as the models grow larger and are trained on more data. This makes evaluation fundamentally different from traditional software testing, where capabilities are explicitly coded and can be tested against pre-defined specifications.

{numref}`emerging-properties` provides a list of emergent abilities of large language models and the scale {cite}`wei2022emergentabilitieslargelanguage`. The relationship between model scale and emergent abilities follows a fascinating non-linear pattern. Below certain size thresholds, specific abilities may be completely absent from the model - it simply cannot perform certain tasks, no matter how much you try to coax them out. However, once the model reaches critical points in its scaling journey, these abilities can suddenly manifest in what researchers call a phase transition - a dramatic shift from inability to capability. This unpredictable emergence of capabilities stands in stark contrast to traditional software development, where features are deliberately implemented and can be systematically tested.

```{figure} ../_static/evals/emerging.png
---
name: emerging-properties
alt: Emerging Properties
class: bg-primary mb-1
scale: 60%
align: center
---
Emergent abilities of large language models and the scale {cite}`wei2022emergentabilitieslargelanguage`.
```

The implications for evaluation are critical. While conventional software testing relies on stable test suites and well-defined acceptance criteria, LLM evaluation must contend with a constantly shifting landscape of capabilities. What worked to evaluate a 7B parameter model may be completely inadequate for a 70B parameter model that has developed new emergent abilities. This dynamic nature of LLM capabilities forces us to fundamentally rethink our approach to testing and evaluation.

## Problem Statement

Consider a practical example that illustrates these challenges: building a Math AI tutoring system for children powered by an LLM. In traditional software development, you would define specific features (like presenting math problems or checking answers) and write tests to verify each function. But with LLMs, you're not just testing predefined features - you're trying to evaluate emergent capabilities like adapting explanations to a child's level, maintaining engagement through conversational learning, and providing age-appropriate safety-bound content.

This fundamental difference raises critical questions about evaluation:
- How do we measure capabilities that weren't explicitly programmed?
- How can we ensure consistent performance when abilities may suddenly emerge or evolve?
- What metrics can capture both the technical accuracy and the subjective quality of responses?

The challenge becomes even more complex when we consider that traditional software evaluation methods simply weren't designed for these kinds of systems. There is an **Evals Gap** between traditional software testing and LLM evaluation. We need new frameworks that can account for both the deterministic aspects we're used to testing and the emergent properties that make LLMs unique. 

{numref}`evals-table` summarizes how LLM evaluation differs from traditional software testing across several key dimensions:
- **Capability Assessment vs Functional Testing**: Traditional software testing validates specific functionality against predefined requirements. LLM evaluation must assess not necessarily pre-defined behavior but also "emergent properties" like reasoning, creativity, and language understanding that extend beyond explicit programming.

- **Metrics and Measurement Challenges**: While traditional software metrics can usually be precisely defined and measured, LLM evaluation often involves subjective qualities like "helpfulness" or "naturalness" that resist straightforward quantification. Even when we try to break these down into numeric scores, the underlying judgment often remains inherently human and context-dependent.

- **Dataset Contamination**: Traditional software testing uses carefully crafted test cases with known inputs and expected outputs (e.g., unit tests). In contrast, LLMs trained on massive internet-scale datasets risk having already seen and memorized evaluation examples during training, which can lead to artificially inflated performance scores. This requires careful dataset curation to ensure test sets are truly unseen by the model and rigorous cross-validation approaches.

- **Benchmark Evolution**: Traditional software maintains stable test suites over time. LLM benchmarks continuously evolve as capabilities advance, making longitudinal performance comparisons difficult and potentially obsoleting older evaluation methods.

- **Human Evaluation Requirements**: Traditional software testing automates most validation. LLM evaluation may demand significant human oversight to assess output quality, appropriateness, and potential biases through structured annotation and systematic review processes.

```{table} Evals of Traditional Software vs LLMs
:name: evals-table
| Aspect                                      | Traditional Software                             | LLMs                                                                                     |
|---------------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------|
| **Capability Assessment**          | Validates specific functionality against requirements | May assess emergent properties like reasoning and creativity                                      |
| **Metrics and Measurement**                             | Precisely defined and measurable metrics                     | Subjective qualities that resist straightforward quantification                                                      |
| **Dataset Contamination**                             | Uses carefully crafted test cases                   | Risk of memorized evaluation examples from training                                                          |
| **Benchmark Evolution**                              | Maintains stable test suites                                 | Continuously evolving benchmarks as capabilities advance                                                 |
| **Human Evaluation**                        | Mostly automated validation                                     | May require significant human oversight                                                        |
```

## Evals Design

First, it's important to make a distinction between evaluating an LLM versus evaluating an LLM-based application. While the former offers foundation capabilities and are typically general-purpose, the latter is more specific and tailored to a particular use case. Here, we define an LLM-based application as a system that uses one or more LLMs to perform a specific task. More specifically, an LLM-based application is the combination of one or more LLM models, their associated prompts and parameters to solve a particular business problem.

That differentiation is important because it changes the scope of evaluation. LLMs are usually evaluated based on their capabilities, which include things like language understanding, reasoning and knowledge. LLM-based applications, instead, should be evaluated based on their end-to-end functionality, performance, and how well they meet business requirements. That distinction has key implications for the design of evaluation systems:

- The same LLM can yield different results in different applications
- Evaluation must align with business objectives
- A great LLM doesn't guarantee a great application!

Examples of key requirements for validation are listed in {numref}`validation-requirements` ranging from Safety, Cognitive, Technical, Meta-Cognitive, to Ethical aspects.

```{table} LLM Application Testing Requirements Matrix
:name: validation-requirements
| Category | Requirement | What to Test | Why It's Important |
|----------|------------|--------------|-------------------|
| Safety | Misinformation Prevention | - Accuracy of factual statements against verified databases<br>- Consistency of responses across similar queries<br>- Rate of fabricated details or hallucinations<br>- Citation and source accuracy<br>- Response behavior with uncertainty<br>- Temporal consistency<br>- Scientific accuracy | - Prevents real-world harm from false information<br>- Maintains user trust<br>- Reduces legal and reputational risks<br>- Ensures reliable decision-making support<br>- Protects against information manipulation |
| Safety | Unqualified Advice | - Recognition of medical, legal, and financial queries<br>- Disclaimer consistency<br>- Professional referral mechanisms<br>- Boundary recognition<br>- Emergency situation handling<br>- Avoidance of specific recommendations | - Prevents harm from incorrect professional advice<br>- Reduces legal liability<br>- Protects vulnerable users<br>- Maintains professional standards<br>- Ensures appropriate expertise utilization |
| Safety | Bias Detection | - Gender, racial, and cultural bias<br>- Demographic representation<br>- Language inclusivity<br>- Stereotype avoidance<br>- Problem-solving fairness<br>- Cultural context awareness | - Prevents reinforcement of societal biases<br>- Ensures equal service quality<br>- Maintains social responsibility<br>- Protects brand reputation<br>- Supports diverse user bases |
| Safety | Privacy Protection | - PII detection and handling<br>- Data anonymization<br>- Information leakage prevention<br>- Context carryover management<br>- Compliance with regulations<br>- Security protocols | - Protects user confidentiality<br>- Ensures regulatory compliance<br>- Maintains data security<br>- Prevents privacy breaches<br>- Safeguards sensitive information |
| Cognitive | Reasoning & Logic | - Multi-step problem-solving<br>- Mathematical computation<br>- Logical fallacy detection<br>- Causal reasoning<br>- Abstract concept handling<br>- Edge case management | - Ensures reliable problem-solving<br>- Maintains computational accuracy<br>- Supports critical thinking<br>- Prevents logical errors<br>- Enables complex decision support |
| Cognitive | Language Understanding | - Context maintenance<br>- Idiom comprehension<br>- Cultural reference accuracy<br>- Sarcasm detection<br>- Technical terminology<br>- Cross-lingual capability | - Ensures effective communication<br>- Prevents misunderstandings<br>- Enables sophisticated interactions<br>- Supports diverse language needs<br>- Maintains conversation quality |
| Technical | Code Generation | - Syntax accuracy<br>- Security vulnerability scanning<br>- Performance optimization<br>- Documentation quality<br>- Error handling<br>- Cross-platform compatibility | - Ensures code reliability<br>- Prevents security issues<br>- Maintains system stability<br>- Supports development efficiency<br>- Reduces technical debt |
| Technical | System Integration | - API handling<br>- Rate limit compliance<br>- Error management<br>- Response time<br>- Resource utilization<br>- Scalability testing | - Ensures system reliability<br>- Maintains performance<br>- Enables scaling<br>- Prevents system failures<br>- Supports integration stability |
| Meta-Cognitive | Self-Awareness | - Knowledge limitation recognition<br>- Uncertainty communication<br>- Correction capabilities<br>- Feedback integration<br>- Edge case recognition<br>- Error acknowledgment | - Builds user trust<br>- Prevents overconfidence<br>- Enables appropriate use<br>- Supports improvement<br>- Maintains reliability |
| Meta-Cognitive | Communication Quality | - Message clarity<br>- Audience appropriateness<br>- Information density<br>- Explanation quality<br>- Summary accuracy<br>- Technical communication | - Ensures understanding<br>- Maintains engagement<br>- Enables knowledge transfer<br>- Builds user satisfaction<br>- Supports effective interaction |
| Ethical | Harmful Content | - Harmful request recognition<br>- Response appropriateness<br>- Content filtering<br>- Emergency handling<br>- User safety protocols<br>- Incident reporting | - Protects user safety<br>- Prevents misuse<br>- Maintains ethical standards<br>- Reduces liability<br>- Ensures responsible use |
| Ethical | Decision-Making | - Moral consistency<br>- Value alignment<br>- Decision fairness<br>- Transparency<br>- Impact assessment<br>- Stakeholder consideration | - Ensures ethical deployment<br>- Maintains standards<br>- Builds trust<br>- Supports values<br>- Prevents harmful impacts |
| Environmental | CO2 Emission | - Energy consumption per request<br>- Model size and complexity impact<br>- Server location and energy sources<br>- Request caching efficiency<br>- Batch processing optimization<br>- Hardware utilization rates<br>- Inference optimization strategies | - Reduces environmental impact<br>- Supports sustainability goals<br>- Optimizes operational costs<br>- Meets environmental regulations<br>- Demonstrates corporate responsibility |
```




### Conceptual Overview

{numref}`conceptual` demonstrates a conceptual design of key components of LLM Application evaluation. 

```{figure} ../_static/evals/conceptual.png
---
name: conceptual
alt: Conceptual Overview
scale: 40%
align: center
---
Conceptual overview of LLM-based application evaluation.
```

We observe three key components:

**1. Examples (Input Dataset):**
- Input:  Query to LLM App, e.g. user message, input file, image, audio, etc.
- Output: A reference expected outcome from the LLM application. Provide ground truth for comparison (*Optional*).
- Purpose: Provides standardized test cases for evaluation.

**2. LLM Application (Processing Layer):**
- Input: Test cases input from Examples
- Output: Generated responses/results
- Purpose: 
  * Represents different LLM implementations/vendors solving a specific task
  * Could be different models (GPT-4, Claude, PaLM, etc.)
  * Could be different configurations of same model
  * Could be different prompting strategies

**3. Evaluator (Assessment Layer):**
- Input: 
  * Outputs from LLM application
  * Reference data from Examples (*Optional*)
- Output: Individual scores for target LLM application
- Purpose:
  * Measures LLM Application performance across defined metrics
  * Applies standardized scoring criteria

Note that **Examples** must provide input data to the LLM Application for further evaluation. However, ground truth data is optional. We will return to this in more detail below, where we will see that ground truth data is not always available or practical. Additionally, there are approaches where one can evaluate LLM Applications without ground truth data.


A more general conceptual design is shown in {numref}`conceptual-multi`, where multiple LLM Applications are evaluated. This design allows for a more comprehensive evaluation of different configurations of LLM-based applications, e.g.:
- Fixing all application parameters and evaluating different LLM models with their default configurations
- Fixing all parameters of an LLM model and evaluating different prompting strategies

```{figure} ../_static/evals/conceptual-multi.svg
---
name: conceptual-multi
alt: Conceptual Overview
scale: 50%
align: center
---
Conceptual overview of Multiple LLM-based applications evaluation.
```

In this evaluation framework, the same inputs are provided to all LLM applications, ensuring that responses are evaluated consistently. Performance is quantified objectively for each LLM Application, and results are ranked for easy comparison. This design leads to two additional components:

**1. Scores (Metrics Layer):**
- Input: Evaluation results from Evaluator
- Output: Quantified performance metrics
- Purpose:
  * Represents performance in numerical form
  * Enables quantitative comparison among LLM applications
  * May include multiple metrics per LLM application

**2. Leaderboard (Ranking Layer):**
- Input: Scores per LLM application
- Output: Ordered ranking of LLMs with scores
- Purpose:
  * Aggregates and ranks performances across LLM applications


### Design Considerations

The design of an LLM application evaluation system depends heavily on the specific use case and business requirements. Here we list important questions for planning an LLM application evaluation system pertaining to each of the key components previously introduced:

**1. Examples (Input Dataset):**
- What types of examples should be included in the test set?
  * Does it cover all important use cases?
  * Are edge cases represented?
  * Is there a good balance of simple and complex examples?
- How do we ensure data quality?
  * Are the examples representative of real-world scenarios?
  * Is there any bias in the test set?
- Should we have separate test sets for different business requirements?
- Do we need human-validated ground truth for all examples?
- Can we use synthetic data to augment the test set?
- How can business updates and user data be reflected in the dataset post-launch?

**2. LLM Applications:**
- What aspects of each LLM app should be standardized for fair comparison?
  * Prompt templates
  * Context length
  * Temperature and other parameters
  * Rate limiting and timeout handling
- What specific configurations impact business requirements?
  * Which LLM application variations should be tested to maximize what we learn?
  * Which LLM capabilities provide the most value for the business and how can we measure that?

**3. Evaluator Design:**
- How do we define success for different types of tasks?
  * Task-specific evaluation criteria
  * Objective metrics vs subjective assessment
- Should evaluation be automated or involve human review?
  * Balance between automation and human judgment
  * Inter-rater reliability for human evaluation
  * Cost and scalability considerations

**4. Scoring System:**
- How should different metrics be weighted?
  * Relative importance of different factors
  * Task-specific prioritization
  * Business requirements alignment
- Should scores be normalized or absolute?
- How to handle failed responses?
- Should we consider confidence scores from the LLMs?

**5. Leaderboard/Ranking:**
- How often should rankings be updated?
- Should ranking include confidence intervals?
- How to handle ties or very close scores?
- Should we maintain separate rankings for different:
  * Business requirements
  * Model Cost Tiers
  * LLM Model Families


Holistically, your evaluation design should be built with scalability in mind to handle growing evaluation needs as the combination of (Input Examples X LLM Applications X Evaluators X Scores X Leaderboards) may grow very fast, particularly for an organization that promotes rapid experimentation and iterative development (good properties!). Finally, one should keep in mind that the evaluation system itself requires validation to confirm its accuracy and reliability vis-a-vis business requirements (evaluating evaluators will be later discussed in this Chapter).

## Metrics

The choice of metric depends on the specific task and desired evaluation criteria. However, one can categorize metrics into two broad categories: **intrinsic** and **extrinsic**.

* **Intrinsic metrics** focus on the model's performance on its primary training objective, which is typically to predict the next token in a sequence.  Perplexity is a common intrinsic metric that measures how well the model predicts a given sample of text.

* **Extrinsic metrics** assess the model's performance on various downstream tasks, which can range from question answering to code generation.  These metrics are not directly tied to the training objective, but they provide valuable insights into the model's ability to generalize to real-world applications.

Here, we are particularly interested in extrinsic metrics, since we are evaluating LLM-based applications rather than base LLM models.

Another way to think about metrics is in terms of the type of the task we evaluate:
1. **Discriminative Task**:
   - Involves distinguishing or classifying between existing data points.
   - Examples: Sentiment analysis, classification, or identifying whether a statement is true or false.
2. **Generative Task**:
   - Involves creating or producing new data or outputs.
   - Examples: Text generation, image synthesis, or summarization.

For discriminative tasks, LLM-based applications may produce log-probabilities or discrete predictions, traditional machine learning metrics like accuracy, precision, recall, and F1 score can be applied. However, generative tasks may output text or images which require different evaluation approaches.

For generative tasks, a range of specialized metrics should be considered. These include match-based metrics such as exact match and prefix match, as well as metrics designed specifically for tasks like summarization and translation, including ROUGE, BLEU, and character n-gram comparisons. The selection of appropriate metrics should align with the specific requirements and characteristics of the task being evaluated.


In {numref}`key-metrics` we provide a short list of widely used extrinsic metrics that can be used to evaluate generative tasks of LLM-based applications, along with their definitions, use cases, and limitations.

```{table} Key Metrics for Evaluating Generative Tasks
:name: key-metrics
| Metric | Definition | Use Case | Limitations |
|--------|------------|----------|-------------|
| **BLEU (Bilingual Evaluation Understudy)** | Measures overlap of n-grams between generated text and reference text | Machine translation and text summarization | - Favors short outputs due to brevity penalty<br>- Insensitive to semantic meaning<br>- Requires high-quality reference texts |
| **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** | Measures overlap between n-grams, words, or sentences of generated text and references, focusing on recall | Text summarization tasks | - Biases toward long outputs<br>- Ignores semantic equivalence<br>- Heavily influenced by reference quality |
| **METEOR (Metric for Evaluation of Translation with Explicit ORdering)** | Considers synonyms, stemming, and paraphrases alongside n-gram overlap | Machine translation, where semantic equivalence matters | - Computationally expensive<br>- Subjective design of synonym/stemming databases |
| **CIDEr (Consensus-based Image Description Evaluation)** | Measures n-gram overlap weighted by TF-IDF, tailored for image captioning | Image caption generation | - Limited applicability outside captioning<br>- Heavily reliant on corpus statistics |
| **TER (Translation Edit Rate)** | Computes number of edits needed to convert hypothesis into reference text | Translation quality evaluation | - Doesn't consider semantic correctness<br>- Penalizes valid paraphrasing |
| **BERTScore** | Uses contextual embeddings from pre-trained BERT to calculate token similarity | Tasks requiring semantic equivalence | - High computational cost<br>- Performance varies with model used |
| **SPICE (Semantic Propositional Image Caption Evaluation)** | Focuses on scene graphs in image captions to evaluate semantic content | Image captioning with emphasis on semantic accuracy | - Designed only for image captions<br>- Less effective in purely textual tasks |
```

A common use case for metrics like BLEU and ROUGE is to evaluate the quality of generated summaries against reference summaries.
As an example, we will demonstrate how to evaluate the quality of Financial Filings summaries against reference summaries (e.g. analyst-prepared highlights). 

We will model our simple metrics-based evaluator with the following components:
- Input: Generated summary and reference summary
- Output: Dictionary with scores for BLEU, ROUGE_1, and ROUGE_2
- Purpose: Evaluate our LLM-based application - Financial Filings summary generator

A *Reference Summary* represents the "ideal" summary. It could be prepared by humans, e.g. expert analysts, or machine-generated. 

In our example, we are particularly interested in evaluating the quality of summaries generated by different (smaller and cheaper) LLM models compared to a *benchmark model* (larger and more expensive). We will use the following setup:
- Benchmark model: `gpt-4o`
- Test models: `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`


First, we define `evaluate_summaries`, a function that calculates BLEU and ROUGE scores to assess text generation quality. It takes a generated summary and reference summary as input, processes them and returns a dictionary with three scores: BLEU (n-gram overlap), ROUGE_1 (unigram comparison), and ROUGE_2 (bigram comparison). This enables quantitative comparison of generated summaries against reference texts. We use HuggingFaces' `evaluate` library to load the metrics.

```bash
pip install evaluate absl-py rouge_score
```


```python
import evaluate
def evaluate_summaries(generated_summary, reference_summary):
    """
    Evaluate generated summaries against reference summaries using multiple metrics.
    
    Args:
        generated_summary (str): The summary generated by the model
        reference_summary (str): The reference/ground truth summary
        
    Returns:
        dict: Dictionary containing scores for different metrics
    """
    # Initialize metrics
    bleu = evaluate.load("google_bleu")
    rouge = evaluate.load("rouge")
    
    # Format inputs for BLEU (expects list of str for predictions and list of list of str for references)
    predictions = [generated_summary]
    references = [reference_summary]
    
    # Compute BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=[references])
    
    # Compute ROUGE scores
    rouge_score = rouge.compute(predictions=predictions, references=references)
    
    # Compute Character metric    
    # Combine all scores into a single dictionary
    scores = {
        'bleu': bleu_score["google_bleu"],
        'rouge1': rouge_score['rouge1'],
        'rouge2': rouge_score['rouge2']
    }
    
    return scores
```

For instance, `evaluate_summaries` can be used to compare two arbitrary sentences and returns a dictionary with our chosen metrics:



```python
sentence1 = "the cat sat on the mat"
sentence2 = "the cat ate the mat"
evaluate_summaries(sentence1, sentence2)

```




    {'bleu': 0.3333333333333333,
     'rouge1': 0.7272727272727272,
     'rouge2': 0.4444444444444445}



Next, we define `generate_summary`, our simple LLM-based SEC filing summirizer application using OpenAI's API. It takes an arbitrary `model`, and an `input` text and returns the corresponding LLM's response with a summary.


```python
from openai import OpenAI
client = OpenAI()

def generate_summary(model, input):
    """
    Generate a summary of input using a given model
    """
    TASK = "Generate a 1-liner summary of the following excerpt from an SEC filing."

    prompt = f"""
    ROLE: You are an expert analyst tasked with summarizing SEC filings.
    TASK: {TASK}
    """
    
    response = client.chat.completions.create(
    model=model,
        messages=[{"role": "system", "content": prompt},
                 {"role": "user", "content": input}]
    )
    return response.choices[0].message.content
```

Now, we define a function `evaluate_summary_models` - our benchmark evaluator - that compares text summaries generated by different language models against a benchmark model. Here's what it does:

1. Takes a benchmark model, list of test models, prompt, and input text
2. Generates a reference summary using the benchmark model and our `generate_summary` function
3. Generates summaries from all test models using `generate_summary` function
4. Evaluates each test model's summary against the benchmark using `evaluate_summaries`
5. Returns evaluation results and the generated summaries


```python
def evaluate_summary_models(model_benchmark, models_test, input):
    """
    Evaluate summaries generated by multiple models
    """
    benchmark_summary = generate_summary(model_benchmark, input)

    # Generate summaries for all test models using list comprehension
    model_summaries = [generate_summary(model, input) for model in models_test]
    
    # Evaluate each model's summary against the benchmark
    evaluation_results = [evaluate_summaries(summary, benchmark_summary) for summary in model_summaries]

    return [evaluation_results, model_summaries, benchmark_summary]
```

We are ready to run our benchmark evaluation. We define a benchmark model and a list of test models and then evaluate each test model's summary against the benchmark. We also print the generated summaries for each model.


```python
model_benchmark = "gpt-4o"
models_test = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
```


```python
evals, model_summaries, benchmark_summary = evaluate_summary_models(model_benchmark, models_test, sec_filing)
```


```python
print(benchmark_summary)

```




    "Apple Inc.'s 10-K filing for the fiscal year ending September 28, 2024, outlines its operational and financial condition, detailing the company's diverse product lines, market activities, and compliance with SEC requirements."




```python
# Print each model name and its summary
for model, summary in zip(models_test, model_summaries):
    print(f"{model}: \n {summary} \n---------------")


```

    gpt-4o-mini: 
     Apple Inc. filed its Annual Report on Form 10-K for the fiscal year ending September 28, 2024, detailing its business operations, risks, and financial condition. 
    ---------------
    gpt-4-turbo: 
     Apple Inc.'s Form 10-K for the fiscal year ended September 28, 2024, details its annual report as a well-known seasoned issuer, confirming compliance with SEC regulations and reporting on stock performances, securities, and corporate governance, while also including forward-looking statements subject to various risks. 
    ---------------
    gpt-3.5-turbo: 
     Apple Inc. filed its Form 10-K with the SEC, revealing financial information for the fiscal year ended September 28, 2024, including details on its products and market performance. 
    ---------------


The benchmark summary from `gpt-4o` provides a balanced overview of the analyzed excerpt from Apple's 10-K filing, focusing on operational status, financial condition, product lines, and regulatory compliance.

When comparing our test models against the benchmark, we observe that:
- `gpt-4o-mini` provides a concise yet comprehensive summary that closely aligns with the benchmark's core message. While it omits product lines, it effectively captures the essential elements of the filing including business operations, risks, and financial condition. Its brevity and focus look (subjectively) similar to our benchmark model.

- `gpt-4-turbo` performs adequately but tends toward verbosity. While it includes relevant information about SEC compliance, it introduces peripheral details about seasoned issuer status and forward-looking statements. The additional complexity makes the summary less focused than gpt-4o-mini's version.

- `gpt-3.5-turbo` looks quite different from the benchmark. Its summary, while factually correct, is overly simplified and misses key aspects of the filing. The model captures basic financial information but fails to convey the breadth of operational and compliance details present in the benchmark summary.

Of course, the above evaluation is only based on a single example and is heavily subjective. It's a "vibe check" on our evaluation results. Now, for an objective analysis, we can look at the quantitative metrics we have chosen and use the `visualize_prompt_comparison` function we write below to visualize the performance of our test models across our predefined quantitative metrics.


```bash
pip install matplotlib
```


```python
def visualize_prompt_comparison(evaluation_results, model_names):
    """
    Create a radar plot comparing different prompt variations
    
    Args:
        evaluation_results (list): List of dictionaries containing evaluation metrics
        model_names (list): List of names for each prompt variation
    """
    from evaluate.visualization import radar_plot
    
    # Format data for visualization
    plot = radar_plot(data=evaluation_results, model_names=model_names)
    return plot
```


```python
# Create and display visualization
plot = visualize_prompt_comparison(evals, models_test)
plot.show()
```

    /tmp/ipykernel_1652501/940173201.py:3: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      plot.show()



    
![png](evals_files/evals_30_1.png)
    


Results demonstrate that tested models perform quite differently on our predefined metrics. The evaluation metrics puts `gpt-4o-mini` as the closest aligned to the benchmark, followed by gpt-4-turbo, and gpt-3.5-turbo showing the largest deviation. This suggests that `gpt-4o-mini` is the best model for this task at least on the metrics we have chosen and for the set of models we have tested.

While evaluating language model outputs inherently involves subjective judgment, establishing a high-quality benchmark model and using quantifiable metrics provide a more objective framework for comparing model performance. This approach transforms an otherwise qualitative assessment into a measurable, data-driven evaluation process.




These metrics provide quantifiable measures of performance, however limitations should be mentioned:

*   **Task-specific nature**:  Chosen set of metrics might not fully capture the nuances of complex generative-based tasks, especially those involving subjective human judgment.
*   **Sensitivity to data distribution**: Performance on these metrics can be influenced by the specific dataset used for evaluation, which might not represent real-world data distribution.
*   **Subjective Acceptable Threshold**: These metrics are not always easy to interpret and set a threshold for (see {cite}`sarmah2024choosethresholdevaluationmetric` for a discussion on how to choose a threshold for an evaluation metric for large language models).
*   **Inability to assess reasoning or factual accuracy**: These metrics primarily focus on surface-level matching and might not reveal the underlying reasoning process of the LLM or its ability to generate factually correct information.

In conclusion, selecting an appropriate extrinsic metrics set depends on the specific task, underlying business requirements and desired evaluation granularity.  Understanding the limitations of these metrics can provide a more comprehensive assessment of LLM performance in real-world applications.

To address these limitations, alternative approaches like **human-based evaluation** and **model-based evaluation** are often used, which will be discussed in the following sections.

## Evaluators

(model-based-eval)=
### Model-Based Evaluation

Traditional metrics like BLEU or ROUGE often fall short in capturing the nuanced, contextual, and creative outputs of LLMs. As an alternative we can consider a "Model-based evaluation" approach. A common approach is to use an LLM as a judge. This is an approach that leverages language models themselves to assess the quality of outputs from other language models. This method involves using a model (often a more capable one) to act as an automated judge, evaluating aspects like accuracy, coherence, and relevance of generated content. Unlike traditional metrics that rely on exact matching or statistical measures, model-based evaluation can capture nuanced aspects of language and provide more contextual assessment. 

As discussed in the paper {cite}`li2024leveraginglargelanguagemodels`, LLM-based evaluation approaches generally fall into two main categories:

1. **Prompt-based evaluation**: This involves using prompts to instruct existing LLMs to evaluate text quality without any fine-tuning. The evaluation can take several forms:
    - Score-based: LLMs assign numerical scores to generated text
    - Probability-based: Using generation probability as a quality metric
    - Likert-style: Rating text quality on discrete scales
    - Pairwise comparison: Directly comparing two texts
    - Ensemble methods: Combining multiple LLM evaluators
2. **Tuning-based evaluation**: This involves fine-tuning open-source LLMs specifically for evaluation tasks. This can be more cost-effective than repeatedly using API calls and allows for domain adaptation.

Once you have chosen your approach, a general LLM-as-a-Judge procedure involves the following steps (see {numref}`llm_judge`):
1. **Define Evaluation Criteria**: Establish clear benchmarks, such as relevance, coherence, accuracy, and fluency.
2. **Prepare Prompts**: Craft effective prompts to guide the LLM in evaluating content against the criteria.
3. **Define Reference Data**: Establish a set of reference data that the judge model can use to evaluate the generated outputs. (*Optional*)
4. **Run Evaluations**: Use the judge model to score outputs. Consider using a large and/or more capable model as a judge to provide more nuanced assessments.
5. **Aggregate and Analyze Results**: Interpret scores to refine applications.

```{figure} ../_static/evals/llm_judge.svg
---
name: llm_judge
alt: Conceptual Overview
scale: 60%
align: center
---
Conceptual overview of LLM-as-a-Judge evaluation.
```

Compared to traditional metrics, LLM-as-a-Judge evaluation offers a more sophisticated assessment framework by leveraging natural language criteria. While metrics focus on statistical measures, judge models excel at evaluating subjective qualities such as creativity, narrative flow, and contextual relevance - aspects that closely mirror human judgment. The judge model processes evaluation guidelines expressed in natural language, functioning similarly to a human reviewer interpreting assessment criteria. One notable consideration is that this approach requires careful prompt engineering to properly define and communicate the evaluation standards to the model.

Prompt Engineering can have a large impact on the quality of the evaluation {cite}`li2024leveraginglargelanguagemodels`. Hence, it's worth noting key prompting best practices when designing LLM-as-a-judge evaluators {cite}`huggingface2024llmjudge`:
1. Use discrete integer scales (e.g., 1-5) rather than continuous ranges 
2. Provide clear rubrics that define what each score level means
3. Include reference answers when available to ground the evaluation
4. Break down complex judgments into specific evaluation criteria

Additionally, the interpretability of the evaluation framework can be fostered by:
1. Requiring explanations and reasoning for scores to increase transparency 
2. Having a hollistic evaluation by considering multiple dimensions such as coherence, relevance, and fluency

Below we provide a sample implementation of an LLM-as-a-Judge evaluation system for our LLM application that generates SEC filing summaries. The code defines:

1. A `JudgeEvaluation` Pydantic model that enforces type validation for four key metrics:
   - Expertise: Rating of analyst-level writing quality
   - Coherence: Score for logical organization
   - Fluency: Assessment of grammar and clarity  
   - Similarity: Measure of alignment with reference text

2. An `evaluate_with_llm()` function that:
   - Takes a judge model, candidate summary, and reference summary as inputs
   - Constructs a detailed prompt instructing the LLM to act as an expert evaluator
   - Uses structured output parsing to return scores in a consistent format
   - Returns scores on a 1-10 scale for each evaluation criterion

The implementation demonstrates how to combine structured data validation with natural language evaluation to create a robust automated assessment system.


```python
from pydantic import BaseModel
from typing import List, Dict

class JudgeEvaluation(BaseModel):
    expertise: int
    coherence: int
    fluency: int
    similarity: int
def evaluate_with_llm(judge_model: str, candidate_summary: str, reference_summary: str) -> Dict[str, float]:
    """
    Use an LLM to evaluate a candidate summary against a reference summary.
    
    Args:
        judge_model (str): Name of the model to use as the judge.
        candidate_summary (str): Generated summary to evaluate.
        reference_summary (str): Ground truth or benchmark summary.
    
    Returns:
        dict: Dictionary containing evaluation scores for specified criteria.
    """
    prompt = f"""
    ROLE: You are an expert evaluator of SEC Filing summaries. Evaluate the following candidate summary against the reference summary on a scale of 1 to 10 for the following criteria:
    - Expertise: Does the summary look like it was written by an expert analyst?
    - Coherence: Is the candidate summary logically organized and easy to understand?
    - Fluency: Is the language of the candidate summary clear and grammatically correct?
    - Similarity: How similar is the candidate summary compared to the reference summary?

    Reference Summary:
    "{reference_summary}"

    Candidate Summary:
    "{candidate_summary}"

    Provide scores in this format:
    Expertise: X, Coherence: Y, Fluency: Z, Similarity: W
    """
    completion = client.beta.chat.completions.parse(
        model=judge_model,
        messages=[{"role": "system", "content": prompt}],
        response_format=JudgeEvaluation
    )
    return completion.choices[0].message.parsed

```

Next, we define a `evaluate_summary_models` function that leverages our LLM-as-a-Judge function to compare summaries generated by different language models. Here's how it works:
   - First, it generates a benchmark summary using the specified benchmark model
   - Then, it generates summaries using each of the test models
   - Finally, it evaluates each test model's summary against the benchmark using the judge model

As a result, we get a list of evaluation results we can use to compare our candidate LLM models across our predefined metrics.



```python

def evaluate_summary_models(judge_model: str, benchmark_model: str, test_models: List[str], input_text: str):
    """
    Evaluate summaries generated by multiple models using an LLM-as-a-Judge approach.
    
    Args:
        judge_model (str): Name of the model to use as the judge.
        benchmark_model (str): Name of the benchmark model.
        test_models (list): List of model names to test.
        input_text (str): Input text for summarization.
    
    Returns:
        tuple: Evaluation results, model summaries, benchmark summary.
    """
    benchmark_summary = generate_summary(benchmark_model, input_text)
    model_summaries = [generate_summary(model, input_text) for model in test_models]

    evaluation_results = [
        evaluate_with_llm(judge_model, summary, benchmark_summary)
        for summary in model_summaries
    ]

    return evaluation_results, model_summaries, benchmark_summary
```


```python
# Example Usage
model_benchmark = "gpt-4o"
models_test = ["gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
judge_model = "gpt-4o"

evals, model_summaries, benchmark_summary = evaluate_summary_models(
    judge_model, model_benchmark, models_test, sec_filing
)
```

Here, we can see the benchmark summary coming from our benchmark model `gpt-4o`:


```python
benchmark_summary
```




    "Apple Inc.'s annual report for the fiscal year ending September 28, 2024, details its business operations, financial condition, and product lines, including iPhones, Macs, iPads, and wearables, and incorporates forward-looking statements regarding its future performance."



Next, we obtain the summaries and evaluation results generated by our test models, `gpt-4o-mini`, `gpt-4-turbo` and `gpt-3.5-turbo`, respectively.


```python
model_summaries
```




    ['Apple Inc. filed its annual Form 10-K report for the fiscal year ended September 28, 2024, detailing its business operations, product lines, and financial performance.',
     "This Form 10-K filing by Apple Inc. for the fiscal year ended September 28, 2024, is an annual report detailing the company's financial performance, including registered securities, compliance with SEC reporting standards, and contains sections on business operations, risk factors, financial data, and management analysis.",
     'Apple Inc., a California-based technology company, reported an aggregate market value of approximately $2.6 trillion held by non-affiliates, with 15.1 billion shares of common stock outstanding as of October 18, 2024.']



As a result we get a list of objects of the Pydantics class we have defined `JudgeEvaluation` which contains the metrics of our evaluation (expertise, coherence, fluency and similarity).


```python
evals
```




    [JudgeEvaluation(expertise=7, coherence=8, fluency=8, similarity=7),
     JudgeEvaluation(expertise=7, coherence=7, fluency=8, similarity=5),
     JudgeEvaluation(expertise=4, coherence=5, fluency=7, similarity=2)]




```python
# Convert evaluation objects to dictionaries
evals_list = [
    {
        "expertise": eval.expertise,
        "coherence": eval.coherence, 
        "fluency": eval.fluency,
        "similarity": eval.similarity
    }
    for eval in evals
]

# Visualize results
plot = visualize_prompt_comparison(evals_list, models_test)
plot.show()

```

    /tmp/ipykernel_1652501/1775618912.py:14: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
      plot.show()



    
![png](evals_files/evals_46_1.png)
    


Looking at the evaluation results across our test models (gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo), we can observe some interesting patterns:

- The `gpt-4o-mini` model performed quite well, achieving high scores across all metrics (expertise: 7, coherence: 8, fluency: 8, similarity: 7). This suggests it maintained good quality while being a smaller model variant of our benchmark model `gpt-4o`.

- The `gpt-4-turbo` model showed similar expertise and fluency (7 and 8 respectively) but slightly lower coherence (7) and notably lower similarity (5) compared to the benchmark. This could indicate some drift from the reference summary while maintaining general quality.

- The `gpt-3.5-turbo` model had the lowest scores overall (expertise: 4, coherence: 5, fluency: 7, similarity: 2), particularly struggling with expertise and similarity to the benchmark. While it maintained reasonable fluency, the significant drop in similarity score suggests substantial deviation from the reference summary.

The visualization helps highlight these differences across models and evaluation dimensions. A clear performance gradient is visible from gpt-4o-mini to gpt-3.5-turbo, with the latter showing marked degradation in most metrics.



Leveraging LLMs for evaluation has several limitations {cite}`li2024leveraginglargelanguagemodels`. Firstly, computational overhead should not be neglected given the inherent cost of running additional model inferences iterations. LLM evaluators can also exhibit various biases, including order bias (preferring certain sequence positions), egocentric bias (favoring outputs from similar models), and length bias. Further, there may be a tight dependency on prompt quality - small prompt variations may lead to substantially different outcomes. It is important to also note challenges around domain-specific evaluation in fields such as medicine, finance, law etc, where a general llm-as-a-judge approach may not be suitable.

The LLM-as-a-Judge strategy can serve as a scalable and nuanced solution to evaluate LLM-based applications. While it does not entirely replace metrics-based or human-based approaches, it significantly augments evaluation workflows, especially in scenarios requiring evaluation of generative outputs. Future improvements in our example include integrating human oversight and refining LLMs for domain-specific evaluation tasks.

One open source solution trying to overcome some of these challenges is Glider {cite}`deshpande2024glidergradingllminteractions`, a 3B evaluator LLM that can score any text input and associated context on arbitrary user defined criteria. Glider is an LLM model trained on 685 domains and 183 criteria whose judgement scores show 91.3% agreement with human judgments, making it suitable for a diverse range of real world applications.



### Evaluating Evaluators

We have discussed how LLMs can be used to evaluate LLM-based aplications. However, how can we evaluate the performance of LLMs that evaluate other LLMs? This is the question that meta evaluation aims to answer. Clearly, the discussion can become quite meta as we need to evaluate the performance of the evaluator to evaluate the performance of the evaluated model. However, one can make a case for two general options:

1. Use a golden-standard dataset that is used to evaluate the performance of LLM evaluators using a "metrics-based" approach.
2. Use a human evaluator to generate reference scores that can be used to evaluate the performance of the LLM evaluator (similar to the human-based evaluation we discussed earlier).

As depicted in {numref}`meta`, the performance of the LLM evaluator can be evaluated by comparing its scores to either a golden-standard dataset or human reference scores. Higher correlation values indicate better performance of the LLM evaluator. For instance, if we were to evaluate the performance of a LLM-as-a-judge evaluator, in the task of evaluating multilingual capability of an LLM:
1. In a "metrics-based" approach, we would first need to define a set of metrics that capture the task of multilingual capability. For instance, we could use the BLEU metric to evaluate the quality of the generated LLM output against a golden dataset (e.g. machine translated text). We would then calculate the correlation between these scores against those generated by the LLM evaluator. The higher the correlation, the better the LLM evaluator.
2. In a "human-based" approach, we would need to recruit human evaluators that are experts in the target languages we are evaluating. Expert humans would provide scores for a set of samples of the input LLM. We would then calculate the correlation between these scores against those generated by the LLM evaluator. The higher the correlation, the better the LLM evaluator.

```{figure} ../_static/evals/meta.png
---
name: meta
alt: Meta Evaluation Conceptual Overview
scale: 30%
align: center
---
Conceptual overview of LLMs Meta Evaluation.
```

An alternative to the above approaches is to use humans to directly evaluate the LLM-judges themselves. A notable example of this is [Judge Arena](https://judgearena.com/) {cite}`judgearena2024`, which is a platform that allows users to vote on which AI model made the better evaluation. Under this approach, the performance of the LLM evaluator is given by the (blind) evaluation of humans who perform the voting on randomly generated pairs of LLM judges as depicted in {numref}`meta2`. Only after submitting a vote, users can see which models were actually doing the judging.

```{figure} ../_static/evals/meta2.png
---
name: meta2
alt: Human-in-the-loop meta evaluation Conceptual Overview
scale: 60%
align: center
---
Human-in-the-loop Meta Evaluation.
```
The LLM input and its prompt are displayed to the human evaluator and are customizable enabling task-specific meta evaluation. Further, the Judge Arena's LLM Judge's prompt is also editable by the user. Its default prompt is presented below:
> Does the model provide relevant and useful responses to the user's needs or questions?
>
> **Scoring Rubric:**
> 
> Score 1: The model's responses are irrelevant or unhelpful to the user's needs or queries.
>
> Score 2: The model sometimes provides helpful information, but often fails to address the user's actual needs or questions.
>
> Score 3: The model generally provides helpful responses that address the user's needs, though it may occasionally miss the mark.
>
> Score 4: The model regularly provides helpful responses that are well-aligned with the user's inquiries, with only rare inaccuracies.
>
> Score 5: The model consistently offers highly relevant and useful responses that perfectly cater to the user's needs and inquiries.

Judge Arena's approach and policy framework has three key benefit worth highlighting:
1. Transparency through open-source code, documentation, and data sharing
2. LLM inclusion criteria requiring scoring/critique capabilities and public accessibility
3. ELO-based leaderboard system with community involvement in evaluations

In that way, the platform enables democratic evaluation of AI judges while maintaining transparency and accessibility standards.

## Benchmarks and Leaderboards

Benchmarks act as standardized tests for LLMs, evaluating their performance across a spectrum of tasks. These tasks simulate real-world applications such as answering questions, generating coherent text, solving mathematical problems, or even writing computer code. They also assess more abstract qualities like fairness, robustness, and cultural understanding.

Benchmarks can be thought as comprehensive "exams" that probe different "subjects" in order to certify an LLM. They help researchers and developers compare models systematically, in a way LLM performance is comparable while enabling the identification of emergent behaviors or capabilities as models evolve in scale and sophistication.

The history of LLM benchmarks reflects the evolving priorities of artificial intelligence research, starting with foundational tasks and moving toward complex, real-world challenges. We can start in 2018 with the introduction of **GLUE** (General Language Understanding Evaluation) {cite}`wang2019gluemultitaskbenchmarkanalysis`, which set a new standard for evaluating natural language understanding. GLUE measured performance on tasks like sentiment analysis and textual entailment, providing a baseline for assessing the fundamental capabilities of language models. Later, **SuperGLUE** {cite}`nangia2019superglue` expanded on this foundation by introducing more nuanced tasks that tested reasoning and language comprehension at a deeper level, challenging the limits of models like BERT and its successors.

As AI capabilities grew, benchmarks evolved to capture broader and more diverse aspects of intelligence. **BIG-Bench** {cite}`srivastava2023imitationgamequantifyingextrapolating` marked a turning point by incorporating over 200 tasks, spanning arithmetic, logic, and creative problem-solving. This collaborative effort aimed to probe emergent abilities in large models, offering insights into how scale and complexity influence performance. Around the same time, specialized benchmarks like **TruthfulQA** {cite}`2021truthfulqa` emerged, addressing the critical need for models to provide accurate and non-deceptive information in a world increasingly dependent on AI for factual content.

**MMLU** (Massive Multitask Language Understanding) {cite}`hendrycks2021measuringmassivemultitasklanguage` launched in 2021, provided a rigorous test of a model’s multidisciplinary knowledge, covering 57 subjects from STEM fields to humanities and social sciences. Similarly, in 2022, Stanford’s **HELM** (Holistic Evaluation of Language Models) {cite}`liang2023holisticevaluationlanguagemodels` set a new standard for multidimensional assessment. HELM expanded the scope of evaluation beyond accuracy, incorporating factors like fairness, robustness, and computational efficiency. This benchmark was designed to address societal concerns surrounding AI, emphasizing safety and inclusion alongside technical performance. 

Specialized benchmarks like **HumanEval** (2021) {cite}`chen2021evaluatinglargelanguagemodels` focused on domain-specific tasks, such as code generation, testing models’ ability to translate natural language descriptions into functional programming code. In contrast, **LMSYS** (2023) brought real-world applicability into focus by evaluating conversational AI through multi-turn dialogues. LMSYS prioritized coherence, contextual understanding, and user satisfaction, providing a practical lens for assessing models like GPT and Claude in dynamic settings.

The **HuggingFace Open LLM** {cite}`openllmleaderboard2024` Leaderboard stands out for its transparency and accessibility in the open-source community. This leaderboard evaluates a wide range of LLMs across diverse tasks, including general knowledge, reasoning, and code-writing. Its commitment to reproducibility ensures that results are verifiable, enabling researchers and practitioners to replicate findings. By focusing on open-source models, it democratizes AI research and fosters innovation across communities, making it a valuable resource for both academics and industry professionals.

The **Chatbot Arena** (2024) Leaderboard (an evolution of LMSYS) {cite}`chiang2024chatbotarenaopenplatform` takes an alternative approach by measuring real-world performance through direct model comparisons. Its evaluation format compares models in live conversations, with human judges providing qualitative assessments. This methodology has gathered hundreds of thousands of human evaluations, offering specific insights into practical model performance. The emphasis on interactive capabilities makes it relevant for developing user-facing applications like virtual assistants and chatbots.

The **AlpacaEval** {cite}`dubois2024lengthcontrolledalpacaevalsimpleway` and **MT-Bench** {cite}`zheng2023judgingllmasajudgemtbenchchatbot` Leaderboards implement automated evaluation using LLMs to assess model performance in multi-turn conversations. This approach enables consistent assessment of dialogue capabilities while reducing human bias. Their methodology measures key aspects of conversational AI, including contextual understanding and response consistency across multiple exchanges.


An important recent development was the release of Global-MMLU {cite}`singh2024globalmmluunderstandingaddressing`, an improved version of MMLU with evaluation coverage across 42 languages. This open dataset, built through collaboration between Argilla, the Hugging Face community, and researchers from leading institutions like Cohere For AI, Mila, MIT, and others, represents a significant step toward more inclusive multilingual LLM evaluation. Hundreds of contributors used Argilla to annotate MMLU questions, revealing that 85% of questions requiring specific cultural knowledge were Western-centric. The newly released dataset is divided into two key subsets: Culturally Agnostic questions that require no specific regional or cultural knowledge, and Culturally Sensitive questions that depend on dialect, cultural, or geographic knowledge. With high-quality translations available for 25 languages, Global-MMLU enables better understanding of LLM capabilities and limitations across different languages and cultural contexts.


A major challenge with these leaderboards and benchmarks is test set contamination - when test data ends up in newer models' training sets, rendering the benchmarks ineffective. While some benchmarks try to address this through crowdsourced prompts and evaluations from humans or LLMs, these approaches introduce their own biases and struggle with difficult questions. **LiveBench** {cite}`white2024livebenchchallengingcontaminationfreellm` represents a novel solution, designed specifically to be resilient to both contamination and evaluation biases. As the first benchmark with continuously updated questions from recent sources, automated objective scoring, and diverse challenging tasks across multiple domains, LiveBench maintains its effectiveness even as models improve. Drawing from recent math competitions, research papers, news, and datasets, it creates contamination-free versions of established benchmark tasks. Current results show even top models achieving considerably lower performance compared to other benchmarks, demonstrating LiveBench's ability to meaningfully differentiate model capabilities with relatively lower saturation. With monthly updates and an open collaborative approach, LiveBench aims to provide sustained value for model evaluation as the field advances.

Another notable benchmark is ZebraLogic {cite}`zebralogic2024`, which evaluates logical reasoning capabilities of LLMs through Logic Grid Puzzles - a type of Constraint Satisfaction Problem {cite}`brailsford1999constraint` commonly found in tests like the LSAT. These puzzles require assigning unique values to N houses across M different features based on given clues, demanding strategic reasoning and deduction to arrive at a unique correct solution. The benchmark's programmatically generated puzzles range from 2x2 to 6x6 in size and test LLMs using one-shot examples with reasoning steps. While humans can solve these puzzles through strategic methods like reductio ad absurdum and elimination, LLMs demonstrate significant limitations in this type of logical reasoning. Even the best-performing model, Claude 3.5 Sonnet, only achieves 33.4% accuracy across all puzzles and 12.4% on hard puzzles, with smaller models (7-10B parameters) solving less than 1% of hard puzzles as of December 2024. These results reveal critical gaps in LLMs' capabilities around counterfactual thinking, reflective reasoning, structured memorization, and compositional generalization.

A significant milestone in AI evaluation came with the launch of the **The Alignment Research Center (ARC) Prize** {cite}`arcprize2024` by ARC Prize Inc., a non-profit for the public advancement of open artificial general intelligence. Hosted by Mike Knoop (Co-founder, Zapier) and François Chollet (Creator of Keras), this prize represents a paradigm shift in how we evaluate language models. Rather than focusing on narrow performance metrics, the ARC Prize assesses what it calls "cognitive sufficiency" - a model's ability to generate meaningful insights and tackle open-ended challenges. This new way to think about LLM evaluation emphasizes creative thinking, sophisticated reasoning, and the capacity to make genuinely useful contributions to human knowledge. Arguably, it is an attempt to define and measure a step towards what it means to achieve AGI (Artificial General Intelligence).


Defining AGI according to ARC Prize:
> Consensus but wrong:
> - AGI is a system that can automate the majority of economically valuable work.

> Correct:
> - AGI is a system that can efficiently acquire new skills and solve open-ended problems.


The ARC benchmark distinguishes itself from other LLM benchmarks especially in its resistance to memorization by prioritizing: 
- Focus on Core Knowledge: Unlike LLM benchmarks that test a broad range of knowledge and skills, often relying heavily on memorization, ARC focuses on core knowledge similar to what a four or five-year-old child might possess. This includes basic concepts like object recognition, counting, and elementary physics.

- Novelty of Tasks: Each ARC puzzle is designed to be novel, meaning it's something you likely wouldn't have encountered before, even if you had memorized the entire internet. This characteristic directly challenges the way LLMs typically operate, which is by leveraging their vast "interpolative memory."

- Emphasis on Program Synthesis: ARC tasks require models to synthesize new solution programs on the fly for each unique puzzle. This stands in contrast to the more common LLM approach of retrieving pre-existing solution programs from memory.

- Resistance to Brute Force Attempts: While acknowledging the possibility, ARC aims to be resistant to brute-force approaches where a model might be trained on millions of similar puzzles to achieve a high score by relying on overlap with the test set.

ARC-AGI tasks are a series of three to five input and output tasks followed by a final task with only the input listed (e.g. {numref}`arc`). Each task tests the utilization of a specific learned skill based on a minimal number of cognitive priors. A successful submission is a pixel-perfect description (color and position) of the final task's output.
```{figure} ../_static/evals/arc.png
---
name: arc
alt: ARC-AGI Task
scale: 50%
align: center
---
Sample ARC-AGI Task.
```

These features make the ARC benchmark a unique test of machine intelligence, focusing on the ability to adapt to novelty and solve problems without relying heavily on memorization. This is more aligned with the concept of general intelligence, which emphasizes the ability to learn efficiently and tackle new challenges.

The ARC-AGI benchmark remained unbeaten for five years as of December 2024 (a minimum score of 85% in the private dataset is required to win) {cite}`arcprizeresults2024`. A key takeaway is that algorithmic improvements, rather than massive computational resources, may be key to exceeding the target score for the ARC-AGI benchmark.


In addition to the benchmarks discussed above, a growing set of domain-specific benchmarks is emerging to help evaluate LLMs in specific verticals, including:
  - FinBench {cite}`zhang2024finbench`: Evaluates LLMs in the financial domain, covering tasks such as terminology understanding, temporal reasoning, future forecasting, scenario planning, and numerical modelling.
  - LegalBench {cite}`guha2023legalbench` : Assesses the legal reasoning abilities of LLMs through tasks crowdsourced by legal professionals
  - Berkeley Function Leaderboard (BFCL) {cite}`patil2023gorilla`: Evaluates LLMs' function-calling abilities


As language models continue to advance in capability and complexity, evaluation frameworks must evolve. Modern benchmarks increasingly incorporate tests for nuanced reasoning, ethical decision-making, and emergent capabilities that weren't previously measurable. This ongoing evolution reflects a deeper understanding that the true value of language models lies not in achieving high scores on standardized tests with narrow task-specific metrics, but in their ability to meaningfully contribute to human understanding and help solve real-world problems while demonstrating the ability to learn and adapt to new tasks.

In the following sections, we will explore some open source tools developers can use to automate and streamline the challenging task of LLMs evals.

## Tools

### LightEval

LightEval {cite}`lighteval` is a lightweight framework for evaluation of LLMs across a variety of standard and bespoke metrics and tasks across multiple inference backends via Python SDK and CLI.

As a motivating example, consider a scenario where financial data has been extracted from SEC financial filings and require econometric analysis. Tasks like estimating autoregressive models for time series forecasting or conducting hypothesis tests on market efficiency are common in financial analysis. Let's evaluate how well different models perform on this type of task.

First, we need to select a benchmark to assess LLMs capabilities in this domain. MMLU has a sub-benchmark called Econometrics we can use for this task. {numref}`mmlu-econometrics` shows a sample of the benchmark dataset from MMLU Econometrics. It consists of multiple-choice questions from econometrics and expected answers.

```{table} MMLU Econometrics Task Dataset sample
:name: mmlu-econometrics
| Question | Options | Correct Options | Correct Options Index | Correct Options Literal |
|-----------|----------|-----------------|---------------------|----------------------|
| Consider the following AR(1) model with the disturbances having zero mean and unit variance: yt = 0.2 + 0.4 yt-1 + ut The (unconditional) mean of y will be given by | ["0.2", "0.4", "0.5", "0.33"] | ["b"] | [3] | ["0.33"] |
| Suppose that a test statistic has associated with it a p-value of 0.08. Which one of the following statements is true? (i) If the size of the test were exactly 8%, we... | ["(ii) and (iv) only", "(i) and (iii) only", "(i), (ii), and (iii) only", "(i), (ii), (iii), and (iv)"] | ["c"] | [2] | ["(i), (ii), and (iii) only"] |
| What would be then consequences for the OLS estimator if heteroscedasticity is present in a regression model but ignored? | ["It will be biased", "It will be inconsistent", "It will be inefficient", "All of (a), (b) and (c) will be true."] | ["c"] | [2] | ["It will be inefficient"] |
| Suppose now that a researcher wishes to use information criteria to determine the optimal lag length for a VAR. 500 observations are available for the bivariate VAR... | ["1 lag", "2 lags", "3 lags", "4 lags"] | ["c"] | [2] | ["3 lags"] |
```

The code sample below demonstrates the LightEval Python SDK framework for evaluating a target LLM model on a given task. First, we instantiate an `EvaluationTracker` which manages result storage, in this example kept in a local directory `output_dir`, and tracks detailed evaluation metrics, optionally pushed to HuggingFace Hub.

Next, we instantiate an object of the class `PipelineParameters` which, in this example, configures the pipeline for parallel processing with a temporary cache in `cache_dir` also setting the maximum number of samples to process to `max_samples`. Then, in `BaseModelConfig` we set up the LLM model we would like to evaluate defined in `pretrained`.

```bash
pip install lighteval[accelerate]
```

```python
import lighteval
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.model_config import BaseModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig
from lighteval.utils.imports import is_accelerate_available
from datetime import timedelta
from accelerate import Accelerator, InitProcessGroupKwargs


def create_evaluation_pipeline(output_dir: str, cache_dir: str, pretrained: str, dtype: str = "float16", max_samples: int = 10, task: str):
    if is_accelerate_available():
        from accelerate import Accelerator, InitProcessGroupKwargs
        accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
    else:
        accelerator = None

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=True,
        push_to_hub=False  
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=EnvConfig(cache_dir=cache_dir),
        override_batch_size=1,
        max_samples=max_samples
    )

    model_config = BaseModelConfig(
        pretrained=pretrained,
        dtype=dtype,
        use_chat_template=True,
        trust_remote_code=True
    )

    pipeline = Pipeline(
        tasks=task,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config
    )
    
    return pipeline
```
{numref}`lighteval` shows a schematic representation of its key components. As inference engine, we leverage `accelerate` for distributed evaluation. `lighteval` also supports other inference backends such as `vllm` and `tgi`.



```{figure} ../_static/evals/lighteval.png
---
name: lighteval
alt: LightEval Python SDK Sample Conceptual Overview.
scale: 35%
align: center
---
LightEval Python SDK Sample Conceptual Overview.
```

This setup allows for systematic evaluation of language model performance on specific tasks while handling distributed computation and result tracking.

The final Pipeline combines these components to evaluate in the user defined `task`, which follows the following format:

```bash
{suite}|{task}|{num_few_shot}|{0 or 1 to automatically reduce `num_few_shot` if prompt is too long}
```

The task string format follows a specific pattern with four components separated by vertical bars (|):

1. suite: The evaluation suite name (e.g., "leaderboard")
2. task: The specific task name (e.g., "mmlu:econometrics") 
3. num_few_shot: The number of few-shot examples to use (e.g., "0" for zero-shot)
4. A binary flag (0 or 1) that controls whether to automatically reduce the number of few-shot examples if the prompt becomes too long

LightEval provides a comprehensive set of evaluation tasks {cite}`lighteval_tasks` and metrics {cite}`lighteval_metrics`. The available tasks  span multiple categories and benchmarks including BigBench, MMLU, TruthfulQA, WinoGrande, and HellaSwag. The framework also supports standard NLP evaluation metrics including BLEU, ROUGE, Exact Match, F1 Score, and Accuracy.

In our case, we choose to evaluate our LLMs on the MMLU econometrics task using zero-shot learning. Hence, we define the `task` as follows:

```python
task = "leaderboard|mmlu:econometrics|0|0"
```

Example usage to evaluate an LLM, for instance `meta-llama/Llama-3.2-1B-Instruct`, on the MMLU econometrics task using zero-shot learning:

```python
task = "leaderboard|mmlu:econometrics|0|0"
model = "meta-llama/Llama-3.2-1B-Instruct"
pipeline = create_evaluation_pipeline(output_dir="./evals/", cache_dir="./cache/", pretrained=model, task=task)
```

We can then evaluate the pipeline, save and show its results as follows:

```python
pipeline.evaluate()
pipeline.save_and_push_results()
pipeline.show_results()
```

The results are then stored in `output_dir` in JSON format.

The same results can be obtained by using the LightEval CLI:

```bash
lighteval accelerate --model_args "pretrained=meta-llama/Llama-3.2-1B-Instruct" --tasks "leaderboard|mmlu:econometrics|0|0" --override_batch_size 1 --output_dir="./evals/"
```

We would like to compare the performance of multiple open source models on the MMLU econometrics task. While we could download and evaluate each model locally, we prefer instead to evaluate them on a remote server to save time and resources. LightEval enables serving the model on a TGI-compatible server/container and then running the evaluation by sending requests to the server {cite}`lighteval_server`. 

For that purpose, we can leverage HuggingFace Serverless Inference API [^lightevalbug] and set a configuration file for LightEval as shown below, where `<MODEL-ID>` is the model identifier on HuggingFace (e.g. `meta-llama/Llama-3.2-1B-Instruct`) and `<HUGGINGFACE-TOKEN>` is the user's HuggingFace API token. Alternatively, you could also pass an URL of a corresponding dedicated inference API if you have one.
[^lightevalbug]: We found a bug in LightEval that prevented it from working with the HuggingFace Serverless Inference API: https://github.com/huggingface/lighteval/issues/422. Thanks to the great work of the LightEval team, this issue has been fixed.
```
model:
  type: "tgi"
  instance:
    inference_server_address: "https://api-inference.huggingface.co/models/<MODEL-ID>"
    inference_server_auth: "<HUGGINGFACE-TOKEN>"
    model_id: null
```

Now we can run the evaluation by sending requests to the server as follows by using the same bash command as before but now setting the `model_config_path` to the path of the configuration file we have just created (e.g. `endpoint_model.yaml`):

```bash
lighteval accelerate --model_config_path="endpoint_model.yaml" --tasks "leaderboard|mmlu:econometrics|0|0" --override_batch_size 1 --output_dir="./evals/"
```

To complete our task, we evaluate a few models from the following model families: `Llama3.2`, `Qwen2.5`, and `SmolLM2` as described in {numref}`model-families`.

```{table} Model Families Evaluated Using LightEval
:name: model-families
| Model Family | Description | Models | References |
|--------------|-------------|---------|------------|
| Llama3.2 Instruct |  LLaMA architecture-based pretrained and instruction-tuned generative models | `Llama-3.2-1B-Instruct` <br> `Llama-3.2-3B-Instruct` | {cite}`meta_llama_models` |
| Qwen2.5 Instruct |  Instruction-tuned LLMs family built by Alibaba Cloud | `Qwen2.5-0.5B-Instruct` <br> `Qwen2.5-1.5B-Instruct`<br> `Qwen2.5-3B-Instruct` | {cite}`gpt2docs,hui2024qwen2,qwen2` |
| SmolLM2 Instruct | Instruction-tuned family of compact language models built by HuggingFace | `SmolLM2-360M-Instruct` <br> `SmolLM2-1.7B-Instruct` | {cite}`allal2024SmolLM2` |
```

We can then compare the performance of these models on the MMLU econometrics task as shown in {numref}`model-comparison`.

```{figure} ../_static/evals/model-comparison.png
---
name: model-comparison
alt: Model Comparison on MMLU Econometrics Task
scale: 50%
align: center
---
Model performance comparison on MMLU Econometrics task, showing accuracy scores across different model sizes and architectures.
```

The results reveal several interesting patterns in model performance. As expected, we observe a trend where larger models consistently achieve higher accuracy scores. The evaluation shows distinct clusters among model families, with Qwen2.5, Llama-3.2, and SmolLM2 each exhibiting their own scaling characteristics, suggesting that architectural differences lead to varying degrees of efficiency as model size increases. Particularly noteworthy is the performance of the Qwen2.5 family, which demonstrates superior accuracy even at smaller model sizes when compared to Llama-3.2. 

Of course, the results should be taken with a grain of salt given the limited size of the dataset (MMLU Econometrics ~ 100), limited number of models and sizes. However, it gives a good indication of the capabilities of the different models tested with Qwen2.5 family being an interesting first candidate as a relatively small yet powerful model demonstrating a good trade-off between performance and size. Once tested on real-world data, the results will change but these initial findings are a good data-driven starting point for model selection as you begin your LLM-based application development.

In summary, LightEval is a simple yet flexible and comprehensive framework for evaluating LLMs across a wide variety of tasks and metrics. It can serve as a first step in selecting your next LLM for a specific task given the exponential growth in number of (open source) models available {cite}`hf_num_models`. Its integration with the Hugging Face ecosystem and modular architecture make it particularly powerful for evaluating open source models. For further details, visit the [official repository](https://github.com/huggingface/lighteval) {cite}`lighteval`.

### LangSmith



Let's revisit our evaluation example when we were interested in evaluating the quality of summaries generated by different (smaller and cheaper) LLM models compared to a benchmark model (larger and more expensive). Recal the setup:

- Benchmark model: gpt-4o

- Test models: gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo


We can run evaluation using only langsmith without the need of langchain.

```bash
!pip uninstall langchain
!pip uninstall langchain-community
!pip uninstall langchain-openai
!pip install langsmith
```

We need to generate an API key to use LangSmith. See instructions [here](https://docs.smith.langchain.com/). Remember to export your API_KEY. Activating tracing will allow us to track logs and foster observability of our evaluation.

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
```


```python
import evaluate as hf_evaluate  # HuggingFace's evaluate
from langsmith import evaluate as langsmith_evaluate  # LangSmith's evaluate
from langsmith import Client
from typing import Dict, Any

ls_client = Client()
```

The code below creates a dataset in LangSmith that will serve as our golden dataset for evaluation. The dataset consists of test cases where we create a single example with the following content:

- An input: Our SEC filing document
- An expected output: A golden summary generated by our benchmark model (`gpt-4o`)

This dataset will allow us to evaluate how well other models perform compared to our benchmark by comparing their generated summaries against these reference summaries. In practice, it's recommended to create a larger dataset with more diverse examples to get a more accurate assessment of model capabilities as well as to estimate confidence intervals for target metrics.



```python
# Define dataset: these are your test cases
dataset_name = "Golden SEC Summary Dataset"
dataset = ls_client.create_dataset(dataset_name)
ls_client.create_examples(
    inputs=[
        {"sec_filing": sec_filing},
    ],
    outputs=[
        {"summary": benchmark_summary},
    ],
    dataset_id=dataset.id,
)
```

Our Dataset is now available in LangSmith as shown in {numref}`langsmith_dataset`.

```{figure} ../_static/evals/langsmith_dataset.png
---
name: langsmith_dataset 
alt: LangSmith Dataset
scale: 25%
align: center
---
LangSmith Dataset
```

Next, we write our evaluator. This evaluator calculates BLEU scores between generated and reference summaries using HuggingFace's evaluate package. The evaluator takes two dictionaries as input - one containing the generated summary and another containing the reference summary. It returns a dictionary with the Google BLEU score, which measures the overlap between n-grams in the generated and reference texts similar to our previous metric-based experiments.


```python
def calculate_scores(outputs: Dict[str, Any], reference_outputs: Dict[str, Any]) -> dict:
    """
    Custom evaluator that calculates BLEU and ROUGE scores between generated and reference summaries
    using HuggingFace's evaluate package
    
    Args:
        outputs (dict): Contains the generated summary
        reference_outputs (dict): Contains the reference summary
    
    Returns:
        dict: Dictionary containing Google BLEU score
    """
    generated = outputs.get("summary", "")
    reference = reference_outputs.get("summary", "")
    
    # Initialize metrics from HuggingFace's evaluate
    bleu = hf_evaluate.load("google_bleu")
    
    # Format inputs for BLEU (expects list of str for predictions and list of list of str for references)
    predictions = [generated]
    references = [reference]
    
    # Compute BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=[references])
    
    return {"key": "google_bleu", "score": bleu_score["google_bleu"]}
```

Now that we have defined our evaluation metrics, let's create a function to generate summaries for our smaller models. The function below takes a dictionary containing the SEC filing text as input and returns a dictionary with the generated summary. The prompt instructs the model to act as an expert analyst and generate a one-line summary of the filing excerpt. We use the same task and model configuration as in our previous experiments to maintain consistency in our evaluation pipeline.



```python
from openai import OpenAI
oai_client = OpenAI()
```


```python
TASK = "Generate a 1-liner summary of the following excerpt from an SEC filing."

PROMPT = f"""
ROLE: You are an expert analyst tasked with summarizing SEC filings.
TASK: {TASK}
"""

xp_model_name = "" # model to be tested

def generate_summary(inputs: dict):
    """
    Generate a summary of input using a given model
    """
    TASK = "Generate a 1-liner summary of the following excerpt from an SEC filing."
    
    response = oai_client.chat.completions.create(
    model=xp_model_name, # model_name is a global variable
        messages=[{"role": "system", "content": PROMPT},
                 {"role": "user", "content": inputs.get("sec_filing")}]
    )
    return {"summary": response.choices[0].message.content}
```


Lastly we define a function to run our evaluation. The `run_evaluation()` function uses LangSmith's `evaluate()` to run evaluations either locally or remotely. When running locally, results are not uploaded to LangSmith's servers. The function takes an application, dataset, and list of evaluators as input and returns the evaluation results. The application is the `generate_summary()` function we would like to evaluate. The `dataset` is the golden summary from the strong model. And we pass a list with our single evaluator `calculate_scores()`. LangSmith also allows for running multiple repetitions of the same experiment to get a more accurate assessment of model capabilities as well as to estimate confidence intervals for target metrics, which we set to 5 repetitions.

This allows us to systematically assess our LLM-based application while maintaining control over where results are stored.


```python
def run_evaluation(app, model_name, dataset,  evaluators, upload_results=False):
    global xp_model_name
    xp_model_name = model_name
    results = langsmith_evaluate(
        app,
        client=None,
        data=dataset,
        evaluators=evaluators,
        experiment_prefix=model_name,
        num_repetitions=5,
        upload_results= upload_results,  # This is the key parameter for local evaluation

    )
    
    return results
```

Now we are ready run evaluation on our app across all target LLM models.


```python
app = generate_summary
```


```python
models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o-mini"]
results = [run_evaluation(app, model, dataset=dataset_name, evaluators=[calculate_scores], upload_results=True) for model in models]

```

    View the evaluation results for experiment: 'gpt-3.5-turbo-386a3620' at:
    https://smith.langchain.com/o/9e1cc3cb-9d6a-4356-ab34-138e0abe8be4/datasets/8741976e-5268-4b75-949f-99477dde5d64/compare?selectedSessions=b831dc1e-90bc-4ed8-8080-fb42444724d6
    
    


    4it [00:10,  2.59s/it]Using the latest cached version of the module from /home/tobias/.cache/huggingface/modules/evaluate_modules/metrics/evaluate-metric--google_bleu/6fc70b7be0088120a372dfdd5d320b39b8bb3630cb8029b193941d9376e86bb0 (last modified on Tue Nov 26 16:50:45 2024) since it couldn't be found locally at evaluate-metric--google_bleu, or remotely on the Hugging Face Hub.
    5it [00:15,  3.09s/it]


    View the evaluation results for experiment: 'gpt-4-turbo-5053784e' at:
    https://smith.langchain.com/o/9e1cc3cb-9d6a-4356-ab34-138e0abe8be4/datasets/8741976e-5268-4b75-949f-99477dde5d64/compare?selectedSessions=64445871-a53c-44b1-a422-4f49b2f9656f
    
    


    5it [00:13,  2.69s/it]


    View the evaluation results for experiment: 'gpt-4o-mini-4b29f3c9' at:
    https://smith.langchain.com/o/9e1cc3cb-9d6a-4356-ab34-138e0abe8be4/datasets/8741976e-5268-4b75-949f-99477dde5d64/compare?selectedSessions=9ef7e39a-2add-410c-89f8-9f1a8b198cf1
    
    


    5it [00:13,  2.61s/it]


We can obtain the results for all experiments including the execution time and the Google BLEU score.


```python
import pandas as pd
```


```python
# Create list of dataframes from results
dfs = [result.to_pandas() for result in results]

for df, model in zip(dfs, models):
    df.insert(0, 'model', model)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>inputs.sec_filing</th>
      <th>outputs.summary</th>
      <th>error</th>
      <th>reference.summary</th>
      <th>feedback.google_bleu</th>
      <th>execution_time</th>
      <th>example_id</th>
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>gpt-3.5-turbo</td>
      <td>UNITED STATES\nSECURITIES AND EXCHANGE COMMISS...</td>
      <td>Apple Inc.'s Form 10-K for the fiscal year end...</td>
      <td>None</td>
      <td>Apple Inc.'s 10-K filing for the fiscal year e...</td>
      <td>0.333333</td>
      <td>1.224388</td>
      <td>feb10f92-3167-41f3-bb1c-d271153a31a8</td>
      <td>5b196b22-9f4c-489c-b020-7823208b42d6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gpt-3.5-turbo</td>
      <td>UNITED STATES\nSECURITIES AND EXCHANGE COMMISS...</td>
      <td>Apple Inc. filed its Form 10-K Annual Report f...</td>
      <td>None</td>
      <td>Apple Inc.'s 10-K filing for the fiscal year e...</td>
      <td>0.348101</td>
      <td>0.722464</td>
      <td>feb10f92-3167-41f3-bb1c-d271153a31a8</td>
      <td>c310f159-064a-4035-97c3-a25bbf43abc2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gpt-3.5-turbo</td>
      <td>UNITED STATES\nSECURITIES AND EXCHANGE COMMISS...</td>
      <td>Apple Inc. filed its annual Form 10-K for the ...</td>
      <td>None</td>
      <td>Apple Inc.'s 10-K filing for the fiscal year e...</td>
      <td>0.386076</td>
      <td>0.704104</td>
      <td>feb10f92-3167-41f3-bb1c-d271153a31a8</td>
      <td>f7f24899-dd50-409e-93cc-6fb1622b60bf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gpt-3.5-turbo</td>
      <td>UNITED STATES\nSECURITIES AND EXCHANGE COMMISS...</td>
      <td>Apple Inc. filed its Annual Report on Form 10-...</td>
      <td>None</td>
      <td>Apple Inc.'s 10-K filing for the fiscal year e...</td>
      <td>0.443038</td>
      <td>0.725059</td>
      <td>feb10f92-3167-41f3-bb1c-d271153a31a8</td>
      <td>242856d6-efb5-4101-b1cf-5805532838ac</td>
    </tr>
    <tr>
      <th>4</th>
      <td>gpt-3.5-turbo</td>
      <td>UNITED STATES\nSECURITIES AND EXCHANGE COMMISS...</td>
      <td>Apple Inc. filed its Annual Report on Form 10-...</td>
      <td>None</td>
      <td>Apple Inc.'s 10-K filing for the fiscal year e...</td>
      <td>0.373418</td>
      <td>0.795302</td>
      <td>feb10f92-3167-41f3-bb1c-d271153a31a8</td>
      <td>ce975169-a0ab-40ce-8e32-efa28d06079d</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Calculate statistics per model
stats = combined_df.groupby('model').agg({
    'feedback.google_bleu': ['mean', 'std'],
    'execution_time': ['mean', 'std']
}).round(4)

# Sort by execution time
stats = stats.sort_values(('execution_time', 'mean'))

# Create a figure with two subplots side by side
import matplotlib.pyplot as plt
import numpy as np

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Define colors for each model
colors = ['#2ecc71', '#3498db', '#e74c3c']
models = stats.index

# Plot for Google BLEU scores
bleu_means = stats[('feedback.google_bleu', 'mean')]
bleu_stds = stats[('feedback.google_bleu', 'std')]

for i, model in enumerate(models):
    ax1.errorbar(i, bleu_means[model], yerr=bleu_stds[model], 
                fmt='o', color=colors[i], markersize=8, capsize=5, 
                label=model)
    if i > 0:
        ax1.plot([i-1, i], [bleu_means[models[i-1]], bleu_means[model]], 
                 '-', color=colors[i], alpha=0.5)

ax1.set_ylabel('Google BLEU Score')
ax1.set_title('Google BLEU Scores by Model')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models, rotation=45)
ax1.set_ylim(bottom=0)  # Set y-axis to start at 0
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot for execution times
exec_means = stats[('execution_time', 'mean')]
exec_stds = stats[('execution_time', 'std')]

for i, model in enumerate(models):
    ax2.errorbar(i, exec_means[model], yerr=exec_stds[model], 
                fmt='o', color=colors[i], markersize=8, capsize=5, 
                label=model)
    if i > 0:
        ax2.plot([i-1, i], [exec_means[models[i-1]], exec_means[model]], 
                 '-', color=colors[i], alpha=0.5)

ax2.set_ylabel('Execution Time (seconds)')
ax2.set_title('Execution Times by Model')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(models, rotation=45)
ax2.set_ylim(bottom=0)  # Set y-axis to start at 0
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display the statistics table
print("\nDetailed Statistics:")
print(stats)

```


    
![png](evals_files/evals_74_0.png)
    


    
    Detailed Statistics:
                  feedback.google_bleu         execution_time        
                                  mean     std           mean     std
    model                                                            
    gpt-4o-mini                 0.4038  0.0453         0.7815  0.0433
    gpt-3.5-turbo               0.3768  0.0424         0.8343  0.2208
    gpt-4-turbo                 0.3519  0.0775         0.9122  0.1482


The evaluation results show interesting differences between the models:

- GPT-3.5-turbo achieved a Google BLEU score of 0.377 (±0.042) with average execution time of 0.83s (±0.22s)
- GPT-4-turbo scored slightly lower at 0.352 (±0.078) and was slower at 0.91s (±0.15s) 
- GPT-4o-mini performed best with a BLEU score of 0.404 (±0.045) while being fastest at 0.78s (±0.04s)

As expected, results suggest that the newer GPT-4o-mini model achieves better quality while maintaining lower latency compared to both GPT-3.5 and GPT-4 turbo variants. The standard deviations indicate that GPT-4-turbo has the most variable output quality, while GPT-4o-mini is most consistent in both quality and speed. Interestingly, the more advanced gpt-4-turbo model has lower BLEU scores but takes longer to execute. This suggests that model size and computational complexity don't necessarily correlate with better performance on this specific summarization task. Of course, this is a very simple task further increasing the number of experiment iterations will yield more accurate results.


Since we decided to upload result, we can also visualize the experiment results in LangSmith as shown in {numref}`langsmith`.

```{figure} ../_static/evals/langsmith.png
---
name: langsmith
alt: LangSmith Experiment Results
scale: 25%
align: center
---
LangSmith Experiment Results
```

### PromptFoo

Promptfoo {cite}`promptfoo2024` is an open-source framework designed for evaluating applications that utilize LLMs. Key features include:

1. **Automated Testing**: Promptfoo provides automated testing capabilities, allowing developers to run custom evaluations tailored to their applications.

2. **Custom Probes**: Developers can create custom probes to focus on specific use cases for instance decoupling prompts from tests cases.

3. **User-Friendly CLI**: The framework features a command-line interface that supports live reloads and caching, facilitating rapid testing and iteration.

We will use promptfoo's command line interface in the following examples. Please follow installation instructions [here](https://www.promptfoo.dev/docs/installation/#for-command-line-usage).


Evals are defined in a configuration file `promptfooconfig.yaml`, which defines elements such as providers, prompts, test cases, and assertions.

In the following example, we will perform a two-step evaluation:

1. Evaluate the performance of different LLM models given a set of constraints.
2. Evaluate the quality of different prompts for the best performing model from 1.


```python
import yaml

# Read the YAML file
with open('promptfoo/model_comparison/promptfooconfig.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Pretty print the YAML content
print(yaml.dump(config, default_flow_style=False, sort_keys=False))

```

    description: Best model eval
    prompts:
    - file://prompt1.txt
    providers:
    - openai:gpt-4o-mini
    - openai:gpt-4
    - openai:gpt-3.5-turbo
    defaultTest:
      assert:
      - type: cost
        threshold: 0.001
      - type: latency
        threshold: 1000
      - type: python
        value: len(output) < 200
      - type: llm-rubric
        value: Does the summary look like it was written by an expert analyst [Yes/No]?
    tests: file://tests.csv
    


The configuration file shows how PromptFoo can be used to evaluate different LLM models. The YAML configuration defines three providers (GPT-4o-mini, GPT-4, and GPT-3.5-turbo) and sets up test assertions to validate their outputs. These assertions check important constraints:

1. Cost efficiency: Each inference must cost less than $0.001
2. Latency requirements: Response time must be under 1000ms 
3. Output length: Generated text must be less than 200 characters
4. Output quality: An LLM-based rubric evaluates if the output appears to be written by an expert (uses openai's gpt-4o model)

The prompts are loaded from an external file (prompt1.txt) and test cases are defined in tests.csv. This structured approach enables systematic evaluation of model performance across multiple decoupled dimensions.


```bash
promptfoo eval --no-cache --output eval.json
```

This command will run the evaluation and store the results in eval.json while making sure that the evaluation is not cached so we are measuring actual latency of the LLMs. The code below processes the PromptFoo evaluation results stored in eval.json. It reads the evaluation data from the JSON file and extracts key metrics including:

- Provider name (e.g. gpt-4, gpt-3.5-turbo)
- Latency in milliseconds 
- Token usage statistics
- Cost per request
- Number of passed/failed assertions
- Prompt token count
- Total number of API requests


```python
import json
import pandas as pd

# Read the JSON file
with open('promptfoo/model_comparison/eval.json', 'r') as f:
    eval_data = json.load(f)

# Extract results into a list of dictionaries
results = []
for prompt in eval_data['results']['prompts']:
    result = {
        'provider': prompt['provider'],
        'latency_ms': prompt['metrics']['totalLatencyMs'],
        'token_usage': prompt['metrics']['tokenUsage']['total'],
        'cost': prompt['metrics']['cost'],
        'assert_pass': prompt['metrics']['assertPassCount'], 
        'assert_fail': prompt['metrics']['assertFailCount'],
        'prompt_tokens': prompt['metrics']['tokenUsage']['prompt'],
        'num_requests': prompt['metrics']['tokenUsage']['numRequests']
    }
    results.append(result)

```


```python
from IPython.display import display, Markdown
```


```python
# Convert to DataFrame
df = pd.DataFrame(results)
print(df)
```

| Provider | Latency (ms) | Token Usage | Cost | Assert Pass | Assert Fail | Prompt Tokens | Num Requests |
|----------|--------------|-------------|------|-------------|-------------|---------------|--------------|
| openai:gpt-4o-mini | 2463 | 97 | $0.000035 | 6 | 2 | 52 | 2 |
| openai:gpt-4 | 3773 | 103 | $0.004620 | 4 | 4 | 52 | 2 |
| openai:gpt-3.5-turbo | 1669 | 95 | $0.000091 | 7 | 1 | 52 | 2 |


The evaluation results reveal interesting performance characteristics across different OpenAI models. GPT-3.5-turbo demonstrates the best overall performance given our criteria with the lowest latency (1669ms), lowest token usage (95), and highest number of passed assertions (7). While GPT-4 shows higher token usage (103) and latency (3773ms), it also has the highest cost per request ($0.00462). The GPT-4-mini variant offers a middle ground, with moderate latency and token usage, while maintaining relatively good assertion performance (6 passes). These results suggest that for this particular evaluation task, GPT-3.5-turbo provides the best balance of performance, reliability, and cost-effectiveness.

Promptfool also offers a web interface for visualizing the evaluation results as shown in {numref}`promptfoo1`. 

```bash
promptfoo view
```

We can observe results per test case (i.e. section of the SEC filing) and per provider. Humans can also manually review the results and provide feedback as well as generate new test cases.

```{figure} ../_static/evals/promptfoo1.png
---
name: promptfoo1
alt: PromptFoo Evaluation Results
scale: 30%
align: center
---
PromptFoo evaluation results showing performance metrics across different models.
```

Now that we have established `GPT-3.5-turbo` as our model of choice given the minimum required criteria based on cost, latency and basic qualitative evaluation, we can compare the performance of different prompts as a next evaluation step. Can we improve the quality of the summaries by using different prompts?

First, we redefine our evaluation criteria. We now would like to select the prompt that delivers the most "detailed" summaries. Our updated promptfoo configuration file is shown below.


```python
# Read the YAML file
with open('promptfoo/prompt_comparison/promptfooconfig.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Pretty print the YAML content
print(yaml.dump(config, default_flow_style=False, sort_keys=False))
```

    description: Best model eval
    prompts:
    - file://prompt1.txt
    - file://prompt2.txt
    - file://prompt3.txt
    providers:
    - openai:gpt-3.5-turbo
    defaultTest:
      assert:
      - type: llm-rubric
        value: 'Evaluate the output based on how detailed it is.  Grade it on a scale
          of 0.0 to 1.0, where:
    
          Score of 0.1: Not much detail.
    
          Score of 0.5: Some detail.
    
          Score of 1.0: Very detailed.
    
          '
    tests: file://tests.csv
    


Note that we are now passing 3 different prompts. And we have updated our assertions to check if the output is "detailed" by leveraging promptfoo's `llm-rubric` assertion which will run an LLM-as-a-Judge for evaluation. Now, let's define 3 prompt variations we would like to test aiming at improving the quality/detail of the summaries.


```python
# Display the prompt variations
from IPython.display import display, Markdown

prompt_files = ['prompt1.txt', 'prompt2.txt', 'prompt3.txt']
prompt_content = []

for file in prompt_files:
    with open(f'promptfoo/prompt_comparison/{file}', 'r') as f:
        content = f.read().strip()
        prompt_content.append(f"### {file}\n---\n{content}\n")

display(Markdown("\n\n".join(prompt_content)))

```


### prompt1.txt
---
'Generate a 1-liner summary of the Section {{section}} from an SEC filing: {{content}}'


### prompt2.txt
---
'ROLE: You are a financial analyst. TASK: Generate a 1-liner summary of the Section {{section}} from an SEC filing: {{content}}'


### prompt3.txt
---
'ROLE: You are a financial analyst. REQUIREMENTS: BE DETAILED. TASK: Generate a 1-liner summary of the Section {{section}} from an SEC filing: {{content}}'



The first prompt matches our previous prompt. The second prompt adds a "financial analyst" role to the prompt. The third prompt expands on second prompt and add a requirement "BE DETAILED".

We can now run the evaluation again.

```bash
promptfoo eval --output eval.json
```


```python
# Read the evaluation results from JSON file
import json
with open('promptfoo/prompt_comparison/eval.json', 'r') as f:
    eval_data = json.load(f)

# Create a list to store the data
data = []

# Extract results for each test case
for result in eval_data['results']['results']:
    section = result['vars']['section']
    prompt_id = result['promptId']
    score = result['gradingResult']['score'] if 'gradingResult' in result else 0.0
    
    # Find the corresponding prompt file
    for prompt in eval_data['results']['prompts']:
        if prompt['id'] == prompt_id:
            prompt_file = prompt['label'].split(':')[0]
            break
            
    # Add to data list
    data.append([section, prompt_file, score])

# Convert to DataFrame
df_raw = pd.DataFrame(data, columns=['Section', 'Prompt', 'Score'])

# Pivot to get desired format
df = df_raw.pivot(index='Section', columns='Prompt', values='Score').reset_index()
df = df[['Section', 'prompt1.txt', 'prompt2.txt', 'prompt3.txt']]

display(Markdown("### Prompt Comparison Results by Section"))
print(df)

```


### Prompt Comparison Results by Section


    Prompt            Section  prompt1.txt  prompt2.txt  prompt3.txt
    0       Legal Proceedings          0.1          0.5          1.0
    1            Risk Factors          0.1          0.5          0.5


The results show that prompt3.txt performs best for Legal Proceedings sections, achieving a perfect score of 1.0 compared to 0.5 for prompt2.txt and 0.1 for prompt1.txt. For Risk Factors sections, both prompt2.txt and prompt3.txt achieve moderate scores of 0.5, while prompt1.txt scores poorly at 0.1. This suggests that prompt3.txt is generally more effective at extracting detailed information, particularly for legal content. In summary, defining a Role and a requirement for the output to be detailed is a good way to improve the quality of the summaries at least for this specific task, model and criteria.


In conclusion, Promptfoo can serve as an effective LLM application evaluation tool particularly for its ability to decouple several components of the evaluation process. Hence enabling the user to focus on the most important aspects of the evaluation given the particular application and criteria making it a valuable and flexible tool for LLM application development.

### Comparison

{numref}`tool-comparison` provides a summarized comparative analysis of three open source frameworks for language models evaluation we have discussed: Lighteval, LangSmith, and Promptfoo. Each framework is assessed based on key features such as integration capabilities, customization options, ease of use, and the ability to facilitate human and LLM collaboration.

```{table} Comparison of Lighteval, LangSmith, and Promptfoo
:name: tool-comparison
| Feature/Aspect       | Lighteval                          | LangSmith                          | Promptfoo                          |
|----------------------|------------------------------------|------------------------------------|------------------------------------|
| **Integration**      | Seamless with Hugging Face models, easy access to multiple inference engines, and remote evaluation (e.g., TGI servers, HF serverless models) | User-provided models, evaluators, and metrics | CLI-based, user-provided models via YAML |
| **Customization**    | Flexible task and metric support, quick evaluation against state-of-the-art leaderboards | Easy setup of custom tasks and metrics with plain vanilla Python functions, lacks predefined tasks and metrics | Default and user-provided probes, metrics, and assertions |
| **Ease of Use**      | User-friendly, minimal setup       | User-friendly, minimal setup, includes UI for result visualization | Simple CLI, rapid testing, includes UI for result visualization |
| **Human/LLM Collaboration**   | Model-based evaluation             | Model-based evaluation  | Supports human and model evaluators       |
```

## Conclusion

Language models have fundamentally transformed how software is developed and evaluated. Unlike conventional systems that produce predictable outputs, LLMs generate varied, probabilistic responses that defy traditional testing approaches. While developers accustomed to deterministic systems may find this shift challenging, continuing to rely on legacy testing methods is unsustainable. These frameworks were not designed to handle the inherent variability of LLM outputs and will ultimately prove inadequate. 

Success requires embracing this new paradigm by implementing comprehensive evals that cover the non-deterministic generative nature of LLMs - this is the new Product Requirements Document (PRD) - and cultivating an organizational mindset focused on iteration, experimentation and growth.

The shift from traditional software testing to LLM evaluation is not just a change in tools but a transformation in mindset. Those who recognize and adapt to this shift will lead the way in harnessing the power of LLMs in software development.


[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC-BY--NC--SA-4.0-lightgrey.svg

```
@misc{tharsistpsouza2024tamingllms,
  author = {Tharsis T. P. Souza},
  title = {Taming LLMs: A Practical Guide to LLM Pitfalls with Open Source Software},
  year = {2024},
  chapter = {The Evals Gap},
  journal = {GitHub repository},
  url = {https://github.com/souzatharsis/tamingLLMs)
}
```
## References
```{bibliography}
:filter: docname in docnames
```


