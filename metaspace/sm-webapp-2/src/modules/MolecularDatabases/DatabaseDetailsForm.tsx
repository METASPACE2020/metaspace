import { defineComponent, reactive } from 'vue'

import { SmForm, PrimaryLabelText, SecondaryLabelText, RadioButton } from '../../components/Form'
import { RichTextArea } from '../../components/RichText'

import { MolecularDB, MolecularDBDetails, UpdateDatabaseDetailsMutation } from '../../api/moldb'
import { formatDatabaseLabel, getDatabaseDetails } from './formatting'
import { ElMessage, ElInput } from '../../lib/element-plus'

interface State {
  model: MolecularDBDetails
  loading: boolean
}

interface Props {
  db: MolecularDB
  submit: (update: UpdateDatabaseDetailsMutation) => void
}

const Details = defineComponent({
  name: 'DatabaseDetailsForm',
  props: {
    db: { type: Object as any, required: true },
    submit: { type: Function, required: true },
  },
  setup(props: Props) {
    const state = reactive<State>({
      model: getDatabaseDetails(props.db),
      loading: false,
    })

    const handleFormSubmit = async () => {
      try {
        state.loading = true
        await props.submit({ id: props.db.id, details: state.model })
        ElMessage({ message: `${formatDatabaseLabel(props.db)} updated`, type: 'success' })
      } catch (e) {
        ElMessage({ message: 'Something went wrong, please try again later', type: 'error' })
      } finally {
        state.loading = false
      }
    }

    return () => (
      <SmForm class="v-rhythm-6" onSubmit={handleFormSubmit}>
        <div>
          <label for="database-full-name">
            <PrimaryLabelText>Full name</PrimaryLabelText>
          </label>
          <ElInput id="database-full-name" v-model={state.model.fullName} />
        </div>
        <RichTextArea
          content={state.model.description}
          onUpdate={(content: string) => {
            if (state.model) {
              state.model.description = content
            }
          }}
        >
          <PrimaryLabelText slot="label">Description</PrimaryLabelText>
        </RichTextArea>
        <div class="radio-wrapper">
          <p class="m-0 mb-3">
            <PrimaryLabelText>Annotation access</PrimaryLabelText>
          </p>
          <RadioButton
            id="database-annotations-private"
            name="isPublic"
            disabled={state.model.isVisible}
            checked={!state.model.isPublic}
            onChange={() => {
              state.model.isPublic = false
              state.model.isVisible = false
            }}
          >
            <PrimaryLabelText>Annotations are private</PrimaryLabelText>
            <SecondaryLabelText>Results will be visible to group members only</SecondaryLabelText>
          </RadioButton>
          <RadioButton
            id="database-annotations-public"
            name="isPublic"
            checked={state.model.isPublic}
            onChange={() => {
              state.model.isPublic = true
            }}
          >
            <PrimaryLabelText>Annotations are public</PrimaryLabelText>
            <SecondaryLabelText>Results will be visible to everyone</SecondaryLabelText>
          </RadioButton>
        </div>
        <div>
          <div class="radio-wrapper">
            <p class="m-0 mb-3">
              <PrimaryLabelText>Custom database access</PrimaryLabelText>
            </p>
            <RadioButton
              id="database-private"
              name="isVisible"
              checked={!state.model.isVisible}
              onChange={() => {
                state.model.isVisible = false
              }}
            >
              <PrimaryLabelText>Custom database is private</PrimaryLabelText>
              <SecondaryLabelText>Custom database will be available for group members only</SecondaryLabelText>
            </RadioButton>
            <RadioButton
              id="database-public"
              name="isVisible"
              checked={state.model.isVisible}
              onChange={() => {
                state.model.isVisible = true
                state.model.isPublic = true
              }}
            >
              <PrimaryLabelText>Custom database is public</PrimaryLabelText>
              <SecondaryLabelText>
                Custom database will be available as annotation option to everyone
              </SecondaryLabelText>
            </RadioButton>
          </div>
          <label for="database-link">
            <PrimaryLabelText>Link</PrimaryLabelText>
          </label>
          <ElInput id="database-link" v-model={state.model.link} />
        </div>
        <RichTextArea
          content={state.model.citation}
          onUpdate={(content: string) => {
            if (state.model) {
              state.model.citation = content
            }
          }}
        >
          <PrimaryLabelText slot="label">Citation</PrimaryLabelText>
        </RichTextArea>
        <button class="el-button el-button--primary">Update details</button>
      </SmForm>
    )
  },
})

export default Details
