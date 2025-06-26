jq -r '
  def re: "[\n\r]+";             # Define the regex first
  def strip: gsub(re; " ");      # Then use it in the strip function

  to_entries[]
  | .value
  | select((.abstract // "" | length > 0) or (.results // "" | length > 0) or (.full_text // "" | length > 0))
  | "\(
      (.title // "" | strip)
    ) \(
      (.abstract // "" | strip)
    )  \(
    )  \(
      (.authors // []
        | map(
            (.affiliation // "" | strip) + " " +
            (.firstname // "" | strip) + " " +
            (.initials // "" | strip) + " " +
            (.lastname // "" | strip)
          )
        | join(", ")
      )
    ) \(
      (.keywords // "" | strip)
    )"
' ../data/vascular_data_pubmed.json > ../data/vascular_data_pubmed.txt
